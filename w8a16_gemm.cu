#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <iostream>


// W8A16 GEMM CUDA kernel
// weight_int8: quantized INT8 weights (1 byte per weight)
// scale: quantization scale factors
// input: FP16 activation
// bias: bias (optional)
__global__ void w8a16_gemm_kernel(
    const uint8_t* __restrict__ weight_int8,   // quantized weights (UINT8, 1 byte per weight)
    const at::Half* __restrict__ scale,        // quantization scale (FP16)
    const at::Half* __restrict__ input,        // input activation (FP16)
    const at::Half* __restrict__ bias,         // bias (FP16, optional)
    at::Half* __restrict__ output,             // output result (FP16)
    int batch_size,
    int in_features,
    int out_features,
    int num_groups,
    int group_size
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= batch_size || col >= out_features) return;

    float sum = 0.0f;

    if (bias != nullptr) {
        sum = __half2float(bias[col]);
    }

    for (int g = 0; g < num_groups; ++g) {
        const int scale_offset = col * num_groups + g;
        const float scale_val = __half2float(scale[scale_offset]);

        for (int i = 0; i < group_size; ++i) {
            const int col_idx = g * group_size + i;
            if (col_idx >= in_features) break;

            const int weight_offset = col * in_features + col_idx;
            const int input_offset = row * in_features + col_idx;

            const uint8_t w_int8 = weight_int8[weight_offset];
            const float w_fp = (static_cast<float>(w_int8) - 128.0f) * scale_val;
            const float x_fp = __half2float(input[input_offset]);

            sum += w_fp * x_fp;
        }
    }

    output[row * out_features + col] = __float2half(sum);
}

// PyTorch call interface
torch::Tensor w8a16_gemm(
    torch::Tensor weight_int8,
    torch::Tensor scale,
    torch::Tensor input,
    torch::Tensor bias,
    int in_features
) {
    const int batch_size = input.size(0);
    const int out_features = weight_int8.size(0);
    const int num_groups = scale.size(1);
    const int group_size = in_features / num_groups;

    auto output = torch::empty({batch_size, out_features}, input.options());

    const dim3 block(16, 16);
    const dim3 grid(
        (out_features + block.x - 1) / block.x,
        (batch_size + block.y - 1) / block.y
    );

    w8a16_gemm_kernel<<<grid, block>>>(
        weight_int8.data_ptr<uint8_t>(),
        scale.data_ptr<at::Half>(),
        input.data_ptr<at::Half>(),
        bias.defined() ? bias.data_ptr<at::Half>() : nullptr,
        output.data_ptr<at::Half>(),
        batch_size,
        in_features,
        out_features,
        num_groups,
        group_size
    );

    cudaDeviceSynchronize();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("w8a16_gemm", &w8a16_gemm, "W8A16 GEMM kernel");
}
