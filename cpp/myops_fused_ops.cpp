// myops_fused_ops.cpp
// Build: icpx -fsycl -O3 -fPIC -shared myops_fused_ops.cpp \
//   -o myops_xpu$(python3 -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))") \
//   $(python3 -m pybind11 --includes) \
//   -I"$TORCH_INC1" -I"$TORCH_INC2" -I"$DPCPP_SYCL_INC" \
//   -L"$TORCH_LIB" -Wl,-rpath,"$TORCH_LIB" \
//   -ldnnl_sycl -ldnnl -ltorch -ltorch_cpu -lc10
//
// Runtime: export SYCL_DEVICE_FILTER=level_zero:gpu
// Make sure LD_LIBRARY_PATH puts $TORCH_LIB first to avoid loader mismatch.

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <unordered_map>
#include <vector>

using namespace dnnl;

// Helpers ---------------------------------------------------------------------
static inline void check_inputs(const torch::Tensor& x,
                                const torch::Tensor& w,
                                const torch::Tensor& b) {
  TORCH_CHECK(
      x.dtype() == torch::kFloat32 || x.dtype() == torch::kBFloat16,
      "x must be float32 or bfloat16");
  TORCH_CHECK(x.dtype() == w.dtype(), "x and w must have the same dtype");
  TORCH_CHECK(x.dtype() == b.dtype(), "x and b must have the same dtype");

  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
  TORCH_CHECK(b.is_contiguous(), "b must be contiguous");

  TORCH_CHECK(x.dim() == 2, "x must be 2D [N,K]");
  TORCH_CHECK(w.dim() == 2, "w must be 2D [M,K]");
  TORCH_CHECK(b.dim() == 1, "b must be 1D [M]");

  const int64_t N = x.size(0);
  const int64_t Kx = x.size(1);
  const int64_t Mw = w.size(0);
  const int64_t Kw = w.size(1);
  TORCH_CHECK(Kx == Kw, "x.size(1) must equal w.size(1) (K)");
  TORCH_CHECK(b.size(0) == Mw, "b.size(0) must equal w.size(0) (M)");

  // Device check (soft): prefer same device & dtype
  TORCH_CHECK(x.device().type() == w.device().type(),
              "x and w must be on the same device");
  TORCH_CHECK(x.device().type() == b.device().type(),
              "x and b must be on the same device");
}

// Core op: y = GELU(x @ w^T + b) ---------------------------------------------
torch::Tensor linear_gelu(const torch::Tensor& x_in,
                          const torch::Tensor& w_in,
                          const torch::Tensor& b_in) {
  check_inputs(x_in, w_in, b_in);

  // Make sure everything is float32 contiguous on the same device
  auto x = x_in.contiguous();
  auto w = w_in.contiguous();
  auto b = b_in.contiguous();

  bool use_bf16 = (x.dtype() == torch::kBFloat16);
  if (use_bf16) {
    x = x.to(torch::kFloat32);
    w = w.to(torch::kFloat32);
    b = b.to(torch::kFloat32);
  }

  const int64_t N = x.size(0);
  const int64_t K = x.size(1);
  const int64_t M = w.size(0); // w: [M,K], we will view as [K,M] (transposed)

  auto y = torch::empty({N, M}, x.options().dtype(torch::kFloat32));

  TORCH_CHECK(x.device().type() == torch::kXPU,
              "linear_gelu fused kernel requires XPU tensors");
  c10::DeviceGuard guard(x.device());
  auto xpu_stream = c10::xpu::getCurrentXPUStream(x.device().index());
  sycl::queue& q = xpu_stream.queue();
  auto dev = q.get_device();
  auto ctx = q.get_context();

  // oneDNN engine/stream bound to this queue
  engine eng = dnnl::sycl_interop::make_engine(dev, ctx);
  stream s = dnnl::sycl_interop::make_stream(eng, q);

  // Memory descriptors with explicit row-major strides:
  // x: [N,K] contiguous -> strides [K,1]
  memory::dims x_dims   {N, K};
  memory::dims x_strides{K, 1};
  auto x_md = memory::desc(x_dims, memory::data_type::f32, x_strides);

  // w: [M,K] contiguous -> we want [K,M] without copy -> use strides [1, K]
  memory::dims w_dims   {K, M};      // logical "transposed" view
  memory::dims w_strides{1, K};
  auto w_md = memory::desc(w_dims, memory::data_type::f32, w_strides);

  // y: [N,M] contiguous -> strides [M,1]
  memory::dims y_dims   {N, M};
  memory::dims y_strides{M, 1};
  auto y_md = memory::desc(y_dims, memory::data_type::f32, y_strides);

  // b: [1,M] broadcast over N
  memory::dims b_dims   {1, M};
  memory::dims b_strides{0, 1};
  auto b_md = memory::desc(b_dims, memory::data_type::f32, b_strides);

  // Post-ops: GELU (erf)
  primitive_attr attr;
  post_ops ops;
  ops.append_eltwise(algorithm::eltwise_gelu_erf, /*alpha=*/0.f, /*beta=*/0.f);
  attr.set_post_ops(ops);

  // Primitive descriptor (matmul with bias)
  // oneDNN 2025.x API: construct directly with engine + attr + all descs
  auto pd = matmul::primitive_desc(eng, x_md, w_md, b_md, y_md, attr);
  auto prim = matmul(pd);

  // Wrap PyTorch XPU memory as USM for oneDNN (no copy)
  auto x_mem = dnnl::sycl_interop::make_memory(
      x_md, eng, dnnl::sycl_interop::memory_kind::usm, x.data_ptr());
  auto w_mem = dnnl::sycl_interop::make_memory(
      w_md, eng, dnnl::sycl_interop::memory_kind::usm, w.data_ptr());
  auto b_mem = dnnl::sycl_interop::make_memory(
      b_md, eng, dnnl::sycl_interop::memory_kind::usm, b.data_ptr());
  auto y_mem = dnnl::sycl_interop::make_memory(
      y_md, eng, dnnl::sycl_interop::memory_kind::usm, y.data_ptr());

  // Execute
  std::unordered_map<int, memory> args{
      {DNNL_ARG_SRC,     x_mem},
      {DNNL_ARG_WEIGHTS, w_mem},
      {DNNL_ARG_BIAS,    b_mem},
      {DNNL_ARG_DST,     y_mem},
  };
  prim.execute(s, args);
  s.wait();

  if (use_bf16) {
    return y.to(torch::kBFloat16);
  }
  return y;
}

static inline memory::format_tag pick_src_tag(bool channels_last) {
  return channels_last ? memory::format_tag::nhwc : memory::format_tag::nchw;
}

static inline memory::format_tag pick_weight_tag(bool channels_last, bool grouped) {
  if (grouped) {
    return channels_last ? memory::format_tag::gohwi : memory::format_tag::goihw;
  }
  return channels_last ? memory::format_tag::ohwi : memory::format_tag::oihw;
}

torch::Tensor conv2d_silu(const torch::Tensor& x_in,
                          const torch::Tensor& w_in,
                          const c10::optional<torch::Tensor>& b_in,
                          std::vector<int64_t> stride,
                          std::vector<int64_t> padding,
                          std::vector<int64_t> dilation,
                          int64_t groups) {
  TORCH_CHECK(x_in.device().type() == torch::kXPU,
              "conv2d_silu fused kernel requires XPU tensors");
  TORCH_CHECK(w_in.device().type() == torch::kXPU,
              "weight must live on XPU");
  TORCH_CHECK(x_in.scalar_type() == w_in.scalar_type(),
              "x and weight must have the same dtype");
  TORCH_CHECK(x_in.scalar_type() == torch::kFloat32 ||
              x_in.scalar_type() == torch::kBFloat16,
              "Only float32 or bfloat16 tensors are supported");

  TORCH_CHECK(stride.size() == 2 && padding.size() == 2 && dilation.size() == 2,
              "stride/padding/dilation must be pairs [H, W]");

  auto dtype = x_in.scalar_type();
  bool channels_last = x_in.is_contiguous(torch::MemoryFormat::ChannelsLast);

  auto x = channels_last ? x_in.contiguous(torch::MemoryFormat::ChannelsLast)
                         : x_in.contiguous();
  auto w = w_in.contiguous();

  const int64_t N = x.size(0);
  const int64_t C = x.size(1);
  const int64_t H = x.size(2);
  const int64_t W = x.size(3);
  const int64_t OC = w.size(0);
  const int64_t IC = w.size(1) * groups;
  const int64_t KH = w.size(2);
  const int64_t KW = w.size(3);

  TORCH_CHECK(C == IC,
              "Input channel mismatch: expected ", IC, " got ", C);

  memory::data_type dt = (dtype == torch::kBFloat16)
                           ? memory::data_type::bf16
                           : memory::data_type::f32;

  auto options = x.options();
  auto bias_tensor = b_in.has_value()
                       ? b_in.value().contiguous()
                       : torch::zeros({OC}, options);
  TORCH_CHECK(bias_tensor.scalar_type() == dtype,
              "bias must match input dtype");
  TORCH_CHECK(bias_tensor.device().type() == torch::kXPU,
              "bias must live on XPU");

  TORCH_CHECK(groups >= 1, "groups must be >= 1");
  TORCH_CHECK(OC % groups == 0,
              "out_channels must be divisible by groups");

  int64_t stride_h = stride[0];
  int64_t stride_w = stride[1];
  int64_t pad_h = padding[0];
  int64_t pad_w = padding[1];
  int64_t dil_h = dilation[0];
  int64_t dil_w = dilation[1];

  auto out_h = (H + 2 * pad_h - dil_h * (KH - 1) - 1) / stride_h + 1;
  auto out_w = (W + 2 * pad_w - dil_w * (KW - 1) - 1) / stride_w + 1;
  TORCH_CHECK(out_h > 0 && out_w > 0, "Invalid output size");

  auto y = torch::empty({N, OC, out_h, out_w}, options,
                        channels_last ? torch::MemoryFormat::ChannelsLast
                                      : torch::MemoryFormat::Contiguous);

  c10::DeviceGuard guard(x.device());
  auto xpu_stream = c10::xpu::getCurrentXPUStream(x.device().index());
  sycl::queue& q = xpu_stream.queue();
  auto dev = q.get_device();
  auto ctx = q.get_context();
  engine eng = dnnl::sycl_interop::make_engine(dev, ctx);
  stream s = dnnl::sycl_interop::make_stream(eng, q);

  memory::dims src_dims{N, C, H, W};
  memory::dims weight_dims = (groups == 1)
                               ? memory::dims{OC, IC / groups, KH, KW}
                               : memory::dims{groups, OC / groups, IC / groups, KH, KW};
  memory::dims dst_dims{N, OC, out_h, out_w};
  memory::dims bias_dims{OC};
  memory::dims stride_dims{stride_h, stride_w};
  memory::dims dilation_dims{dil_h - 1, dil_w - 1};
  memory::dims padding_dims{pad_h, pad_w};

  auto src_md = memory::desc(src_dims, dt, pick_src_tag(channels_last));
  auto weight_md = memory::desc(weight_dims, dt,
                                pick_weight_tag(channels_last, groups > 1));
  auto dst_md = memory::desc(dst_dims, dt, pick_src_tag(channels_last));
  auto bias_md = memory::desc(bias_dims, dt, memory::format_tag::x);

  primitive_attr attr;
  post_ops ops;
  ops.append_eltwise(algorithm::eltwise_swish, 1.f, 0.f);
  attr.set_post_ops(ops);

  auto conv_pd = convolution_forward::primitive_desc(
      eng,
      prop_kind::forward_inference,
      algorithm::convolution_direct,
      src_md,
      weight_md,
      bias_md,
      dst_md,
      stride_dims,
      dilation_dims,
      padding_dims,
      padding_dims,
      attr);

  auto src_mem = dnnl::sycl_interop::make_memory(
      src_md, eng, dnnl::sycl_interop::memory_kind::usm, x.data_ptr());
  auto weight_mem = dnnl::sycl_interop::make_memory(
      weight_md, eng, dnnl::sycl_interop::memory_kind::usm, w.data_ptr());
  auto bias_mem = dnnl::sycl_interop::make_memory(
      bias_md, eng, dnnl::sycl_interop::memory_kind::usm, bias_tensor.data_ptr());
  auto dst_mem = dnnl::sycl_interop::make_memory(
      dst_md, eng, dnnl::sycl_interop::memory_kind::usm, y.data_ptr());

  std::unordered_map<int, memory> args{
      {DNNL_ARG_SRC, src_mem},
      {DNNL_ARG_WEIGHTS, weight_mem},
      {DNNL_ARG_BIAS, bias_mem},
      {DNNL_ARG_DST, dst_mem},
  };

  auto conv = convolution_forward(conv_pd);
  conv.execute(s, args);
  s.wait();

  return y;
}

// Python binding --------------------------------------------------------------
PYBIND11_MODULE(myops_xpu, m) {
  m.def("linear_gelu", &linear_gelu,
        "Fused Linear + Bias + GELU (oneDNN/SYCL) using XPU memory (zero-copy).");
  m.def("conv2d_silu", &conv2d_silu,
        "Fused Conv2d + Bias + SiLU (oneDNN/SYCL)",
        pybind11::arg("x"),
        pybind11::arg("weight"),
        pybind11::arg("bias") = c10::optional<torch::Tensor>{},
        pybind11::arg("stride") = std::vector<int64_t>{1, 1},
        pybind11::arg("padding") = std::vector<int64_t>{0, 0},
        pybind11::arg("dilation") = std::vector<int64_t>{1, 1},
        pybind11::arg("groups") = 1);
}
