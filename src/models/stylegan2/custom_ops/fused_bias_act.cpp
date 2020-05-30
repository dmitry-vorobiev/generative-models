#include <torch/extension.h>

// CUDA declarations

torch::Tensor fused_bias_act_op(
    const torch::Tensor& x, 
    const torch::Tensor& b,
    const torch::Tensor& ref, 
    int grad,
    int axis,
    int act,
    float alpha,
    float gain);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_NON_NEGATIVE(x) TORCH_CHECK(x >= 0, #x " must be non-negative")

torch::Tensor fused_bias_act(
    const torch::Tensor& x, 
    const torch::Tensor& b,
    const torch::Tensor& ref, 
    int grad,
    int axis,
    int act,
    float alpha,
    float gain)
{
    CHECK_INPUT(x);
    CHECK_INPUT(b);
    CHECK_INPUT(ref);

    CHECK_NON_NEGATIVE(act);
    CHECK_NON_NEGATIVE(grad);
    CHECK_NON_NEGATIVE(axis);

    TORCH_CHECK(b.numel() == 0 || axis < x.dim(), "axis out of bounds");
    TORCH_CHECK(b.dim() == 1, "b must have rank 1");
    TORCH_CHECK(b.numel() == 0 || b.numel() == x.size(axis), "b has wrong number of elements");
    TORCH_CHECK(ref.numel() == ((grad == 0) ? 0 : x.numel()), "ref has wrong number of elements");

    return fused_bias_act_op(x, b, ref, grad, axis, act, alpha, gain);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using namespace py;
    
    m.def("call", &fused_bias_act, "Fused ops: y = act(x + b) * gain (CUDA)", 
    arg("x"), 
    arg("b"), 
    arg("ref"), 
    arg("grad")  = 0, 
    arg("axis")  = 1, 
    arg("act")   = 0, 
    arg("alpha") = 0.0, 
    arg("gain")  = 1.0);
}
