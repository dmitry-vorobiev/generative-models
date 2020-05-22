#include <torch/extension.h>

// CUDA declarations

torch::Tensor upfirdn_2d_op(
    torch::Tensor input, 
    torch::Tensor kernel, 
    int upx, 
    int upy, 
    int downx, 
    int downy,
    int padx0, 
    int padx1, 
    int pady0, 
    int pady1);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_NDIM(x, n) TORCH_CHECK(x.dim() == n, #x " must have rank " #n)
#define CHECK_MIN_SIZE(x, n) TORCH_CHECK(x.size(0) >= n && x.size(1) >= n, #x " must be at least " #n "x" #n)

torch::Tensor upfirdn_2d(
    torch::Tensor input, 
    torch::Tensor kernel, 
    int upx, 
    int upy, 
    int downx, 
    int downy,
    int padx0, 
    int padx1, 
    int pady0, 
    int pady1) 
{
    CHECK_INPUT(input);
    CHECK_INPUT(kernel);

    CHECK_NDIM(input, 4);
    CHECK_NDIM(kernel, 2);

    CHECK_MIN_SIZE(kernel, 1);

    return upfirdn_2d_op(input, kernel, upx, upy, downx, downy, padx0, padx1, pady0, pady1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &upfirdn_2d, "upfirdn_2d (CUDA)");
}