#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

template <typename T>
__device__ __forceinline__ T gelu_tanh(T x) {
  const T kAlpha = T(0.7978845608028654);
  const T kBeta  = T(0.044715);
  T x3 = x * x * x;
  T u  = kAlpha * (x + kBeta * x3);
  T t  = tanh(u);
  return T(0.5) * x * (T(1) + t);
}

template <typename T>
__device__ __forceinline__ T dgelu_tanh(T x) {
  const T kAlpha = T(0.7978845608028654);
  const T kBeta  = T(0.044715);
  T x2 = x * x;
  T u  = kAlpha * (x + kBeta * x2 * x);
  T t  = tanh(u);
  T sech2 = T(1) - t * t;
  return T(0.5) * (T(1) + t) + T(0.5) * x * sech2 * kAlpha * (T(1) + T(3) * kBeta * x2);
}

template <typename T>
__global__ void bias_gelu_fwd_kernel(const T* __restrict__ in, const T* __restrict__ bias,
                                     T* __restrict__ out, int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int size = rows * cols;
  if (idx >= size) return;
  int j = idx % cols;
  T x = in[idx] + bias[j];
  out[idx] = gelu_tanh(x);
}

template <typename T>
__global__ void bias_gelu_bwd_kernel(const T* __restrict__ grad_out, const T* __restrict__ x_plus_b,
                                     T* __restrict__ grad_xplusb, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) return;
  T g = grad_out[idx];
  T x = x_plus_b[idx];
  grad_xplusb[idx] = g * dgelu_tanh(x);
}

} // namespace

torch::Tensor bias_gelu_forward_cuda(torch::Tensor x, torch::Tensor bias) {
  auto M = x.size(0);
  auto N = x.size(1);
  auto x_contig = x.contiguous();
  auto b_contig = bias.contiguous();
  auto y = torch::empty_like(x_contig);

  int size = static_cast<int>(M * N);
  int threads = 256;
  int blocks  = (size + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(x_contig.scalar_type(), "bias_gelu_forward", [&](){
    bias_gelu_fwd_kernel<scalar_t><<<blocks, threads>>>(
      x_contig.data_ptr<scalar_t>(),
      b_contig.data_ptr<scalar_t>(),
      y.data_ptr<scalar_t>(),
      static_cast<int>(M), static_cast<int>(N)
    );
  });
  return y;
}

torch::Tensor bias_gelu_backward_cuda(torch::Tensor grad_out, torch::Tensor x_plus_b) {
  auto go = grad_out.contiguous();
  auto xb = x_plus_b.contiguous();
  auto grad_xb = torch::empty_like(go);

  int size = static_cast<int>(go.numel());
  int threads = 256;
  int blocks  = (size + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(go.scalar_type(), "bias_gelu_backward", [&](){
    bias_gelu_bwd_kernel<scalar_t><<<blocks, threads>>>(
      go.data_ptr<scalar_t>(),
      xb.data_ptr<scalar_t>(),
      grad_xb.data_ptr<scalar_t>(),
      size
    );
  });
  return grad_xb;
}

// ============================
// [ADD] Conv2d + Bias + SiLU
//  - Input: preact [N,C,H,W]  (conv output without bias)
//  - Bias : [C]
//  - Output: y = silu(preact + bias[c])
//  - Backward returns grad_xplusb so conv/autograd가 나머지 grad 계산
// ============================
template <typename T>
__device__ __forceinline__ T _sigmoid(T x) {
  return T(1) / (T(1) + exp(-x));
}

template <typename T>
__device__ __forceinline__ T _dsilu(T x) {
  // d/dx (x * sigmoid(x)) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
  T s = _sigmoid(x);
  return s * (T(1) + x * (T(1) - s));
}

template <typename T>
__global__ void bias_silu_fwd_kernel(
    const T* __restrict__ in, const T* __restrict__ bias,
    T* __restrict__ out, int N, int C, int H, int W) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int size = N * C * H * W;
  if (idx >= size) return;
  int spatial = H * W;
  int c = (idx / spatial) % C;
  T x = in[idx] + bias[c];
  out[idx] = x * _sigmoid(x);
}

template <typename T>
__global__ void bias_silu_bwd_kernel(
    const T* __restrict__ grad_out, const T* __restrict__ in, const T* __restrict__ bias,
    T* __restrict__ grad_xb, int N, int C, int H, int W) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int size = N * C * H * W;
  if (idx >= size) return;
  int spatial = H * W;
  int c = (idx / spatial) % C;
  T x = in[idx] + bias[c];
  grad_xb[idx] = grad_out[idx] * _dsilu(x);
}

torch::Tensor conv_bias_silu_forward_cuda(torch::Tensor preact, torch::Tensor bias) {
  TORCH_CHECK(preact.dim()==4, "preact must be NCHW");
  auto N = (int)preact.size(0), C = (int)preact.size(1), H = (int)preact.size(2), W = (int)preact.size(3);
  auto x = preact.contiguous();
  auto b = bias.contiguous();
  auto y = torch::empty_like(x);
  int size = N*C*H*W, threads=256, blocks=(size+threads-1)/threads;
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_bias_silu_forward", [&](){
    bias_silu_fwd_kernel<scalar_t><<<blocks,threads>>>(
      x.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), N,C,H,W);
  });
  return y;
}

torch::Tensor conv_bias_silu_backward_cuda(torch::Tensor grad_out, torch::Tensor preact, torch::Tensor bias) {
  TORCH_CHECK(preact.dim()==4, "preact must be NCHW");
  auto N = (int)preact.size(0), C = (int)preact.size(1), H = (int)preact.size(2), W = (int)preact.size(3);
  auto go = grad_out.contiguous();
  auto x  = preact.contiguous();
  auto b  = bias.contiguous();
  auto grad_xb = torch::empty_like(go);
  int size = N*C*H*W, threads=256, blocks=(size+threads-1)/threads;
  AT_DISPATCH_FLOATING_TYPES(go.scalar_type(), "conv_bias_silu_backward", [&](){
    bias_silu_bwd_kernel<scalar_t><<<blocks,threads>>>(
      go.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
      grad_xb.data_ptr<scalar_t>(), N,C,H,W);
  });
  return grad_xb;
}
