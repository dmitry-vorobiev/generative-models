// Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/stylegan2/license.html

#define EIGEN_USE_GPU
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>


#define OP_CHECK_CUDA_ERROR(CUDA_CALL) do { cudaError_t err = CUDA_CALL; TORCH_CHECK(err == cudaSuccess, cudaGetErrorName(err)); } while (false)

//------------------------------------------------------------------------
// CUDA kernel.

template <class T>
struct FusedBiasActKernelParams
{
    const T*    x;      // [sizeX]
    const T*    b;      // [sizeB] or NULL
    const T*    ref;    // [sizeX] or NULL
    T*          y;      // [sizeX]

    int         grad;
    int         axis;
    int         act;
    float       alpha;
    float       gain;

    int         sizeX;
    int         sizeB;
    int         stepB;
    int         loopX;
};

template <class T>
static __global__ void FusedBiasActKernel(const FusedBiasActKernelParams<T> p)
{
    const float expRange        = 80.0f;
    const float halfExpRange    = 40.0f;
    const float seluScale       = 1.0507009873554804934193349852946f;
    const float seluAlpha       = 1.6732632423543772848170429916717f;

    // Loop over elements.
    int xi = blockIdx.x * p.loopX * blockDim.x + threadIdx.x;
    for (int loopIdx = 0; loopIdx < p.loopX && xi < p.sizeX; loopIdx++, xi += blockDim.x)
    {
        // Load and apply bias.
        float x = (float)p.x[xi];
        if (p.b)
            x += (float)p.b[(xi / p.stepB) % p.sizeB];
        float ref = (p.ref) ? (float)p.ref[xi] : 0.0f;
        if (p.gain != 0.0f & p.act != 9)
            ref /= p.gain;

        // Evaluate activation func.
        float y;
        switch (p.act * 10 + p.grad)
        {
            // linear
            default:
            case 10: y = x; break;
            case 11: y = x; break;
            case 12: y = 0.0f; break;

            // relu
            case 20: y = (x > 0.0f) ? x : 0.0f; break;
            case 21: y = (ref > 0.0f) ? x : 0.0f; break;
            case 22: y = 0.0f; break;

            // lrelu
            case 30: y = (x > 0.0f) ? x : x * p.alpha; break;
            case 31: y = (ref > 0.0f) ? x : x * p.alpha; break;
            case 32: y = 0.0f; break;

            // tanh
            case 40: { float c = expf(x); float d = 1.0f / c; y = (x < -expRange) ? -1.0f : (x > expRange) ? 1.0f : (c - d) / (c + d); } break;
            case 41: y = x * (1.0f - ref * ref); break;
            case 42: y = x * (1.0f - ref * ref) * (-2.0f * ref); break;

            // sigmoid
            case 50: y = (x < -expRange) ? 0.0f : 1.0f / (expf(-x) + 1.0f); break;
            case 51: y = x * ref * (1.0f - ref); break;
            case 52: y = x * ref * (1.0f - ref) * (1.0f - 2.0f * ref); break;

            // elu
            case 60: y = (x >= 0.0f) ? x : expf(x) - 1.0f; break;
            case 61: y = (ref >= 0.0f) ? x : x * (ref + 1.0f); break;
            case 62: y = (ref >= 0.0f) ? 0.0f : x * (ref + 1.0f); break;

            // selu
            case 70: y = (x >= 0.0f) ? seluScale * x : (seluScale * seluAlpha) * (expf(x) - 1.0f); break;
            case 71: y = (ref >= 0.0f) ? x * seluScale : x * (ref + seluScale * seluAlpha); break;
            case 72: y = (ref >= 0.0f) ? 0.0f : x * (ref + seluScale * seluAlpha); break;

            // softplus
            case 80: y = (x > expRange) ? x : logf(expf(x) + 1.0f); break;
            case 81: y = x * (1.0f - expf(-ref)); break;
            case 82: { float c = expf(-ref); y = x * c * (1.0f - c); } break;

            // swish
            case 90: y = (x < -expRange) ? 0.0f : x / (expf(-x) + 1.0f); break;
            case 91: { float c = expf(ref); float d = c + 1.0f; y = (ref > halfExpRange) ? x : x * c * (ref + d) / (d * d); } break;
            case 92: { float c = expf(ref); float d = c + 1.0f; y = (ref > halfExpRange) ? 0.0f : x * c * (ref * (2.0f - d) + 2.0f * d) / (d * d * d); } break;
        }

        // Apply gain and store.
        p.y[xi] = (T)(y * p.gain);
    }
}

//------------------------------------------------------------------------
// PyTorch op.
torch::Tensor fused_bias_act_op(
    const torch::Tensor& x, 
    const torch::Tensor& b,
    const torch::Tensor& ref, 
    int grad,
    int axis,
    int act,
    int alpha,
    int gain)
{
    OP_CHECK_CUDA_ERROR(cudaSetDevice(x.get_device()));

    cudaStream_t stream;
    OP_CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    auto y = at::empty_like(x);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "fused_bias_act_closure", [&] {
        FusedBiasActKernelParams<scalar_t> p;

        p.x = x.data_ptr<scalar_t>();
        p.b = (b.numel()) ? b.data_ptr<scalar_t>() : nullptr;
        p.ref = (ref.numel()) ? ref.data_ptr<scalar_t>() : nullptr;
        p.y = y.data_ptr<scalar_t>();

        p.sizeX = (int)x.numel();
        p.sizeB = (int)b.numel();
        p.stepB = 1;
        for (int i = axis + 1; i < x.dim(); i++)
            p.stepB *= (int)x.size(i);

        p.loopX = 4;
        int blockSize = 4 * 32;
        int gridSize = (p.sizeX - 1) / (p.loopX * blockSize) + 1;
        void* args[] = {&p};
        OP_CHECK_CUDA_ERROR(cudaLaunchKernel((void*)FusedBiasActKernel<scalar_t>, gridSize, blockSize, args, 0, stream));
    });

    OP_CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    return y;
}
