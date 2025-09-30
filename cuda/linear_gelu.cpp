#include <torch/extension.h>

torch::Tensor bias_gelu_forward_cuda(torch::Tensor x, torch::Tensor bias);
torch::Tensor bias_gelu_backward_cuda(torch::Tensor grad_out, torch::Tensor x_plus_b);

torch::Tensor bias_gelu_forward(torch::Tensor x, torch::Tensor bias) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
  return bias_gelu_forward_cuda(x, bias);
}

torch::Tensor bias_gelu_backward(torch::Tensor grad_out, torch::Tensor x_plus_b) {
  TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
  TORCH_CHECK(x_plus_b.is_cuda(), "x_plus_b must be CUDA");
  return bias_gelu_backward_cuda(grad_out, x_plus_b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bias_gelu_forward", &bias_gelu_forward, "Bias+GELU forward (CUDA)");
  m.def("bias_gelu_backward", &bias_gelu_backward, "Bias+GELU backward (CUDA)");
}

// [ADD] conv + bias + silu
torch::Tensor conv_bias_silu_forward_cuda(torch::Tensor preact, torch::Tensor bias);
torch::Tensor conv_bias_silu_backward_cuda(torch::Tensor grad_out, torch::Tensor preact, torch::Tensor bias);

torch::Tensor conv_bias_silu_forward(torch::Tensor preact, torch::Tensor bias){
  TORCH_CHECK(preact.is_cuda() && bias.is_cuda(), "inputs must be CUDA");
  return conv_bias_silu_forward_cuda(preact, bias);
}
torch::Tensor conv_bias_silu_backward(torch::Tensor grad_out, torch::Tensor preact, torch::Tensor bias){
  TORCH_CHECK(grad_out.is_cuda() && preact.is_cuda() && bias.is_cuda(), "inputs must be CUDA");
  return conv_bias_silu_backward_cuda(grad_out, preact, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // ... 기존 바인딩들 ...
  m.def("conv_bias_silu_forward", &conv_bias_silu_forward, "ConvBias+SiLU forward (CUDA)");
  m.def("conv_bias_silu_backward", &conv_bias_silu_backward, "ConvBias+SiLU backward (CUDA)");
}
