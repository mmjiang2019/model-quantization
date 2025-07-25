import torch
import torch.nn.functional as F

from model_quantize_and_dequantize.Deepth.Quantize_and_Dequantize.symmetric_quantize_and_dequantize import (
    linear_quantize_symmetric,
    )

# w8a32 means weights in 8-bits and activations in 32-bits.
def quantized_linear_w8a32_without_bias(input: torch.Tensor, q_w: torch.Tensor, s_w, z_w):
    assert input.dtype == torch.float32, "input must be float32"
    assert q_w.dtype == torch.int8, "q_w must be int8"

    dequantized_weight = q_w.to(torch.float32) * (s_w - z_w)

    output = F.linear(input, dequantized_weight)
    return output

if __name__ == '__main__':
    input = torch.tensor([1, 2, 3], dtype=torch.float32)
    weight = torch.tensor([[-2,   -1.13, 0.42],
                       [-1.51, 0.25, 1.62],
                       [0.23,  1.35, 2.15]])
    
    q_w, s_w = linear_quantize_symmetric(weight)

    output = quantized_linear_w8a32_without_bias(input, q_w, s_w, 0)
    print(f"ouput with w8a32 quantization without bias:\n{output}")

    output = F.linear(input, weight)
    print(f"\noutput without quantization and bias:\n{output}")