import torch

from model_quantize_and_dequantize.Deepth.helper.utils import plot_quantization_errors

# The formula is:
#   r = s * (q - z)
# where:
# 1. r is the original tensor
# 2. q is the quantized tensor
# 3. s is the scale with same type as r
# 3. z is the zero point with same type as q
# Here we implement the formula directly with random scale and zero point given by the user
def linear_quantize_with_scale_and_zero_point(
        tensor : torch.Tensor,
        scale,
        zero_point,
        dtype: torch.dtype = torch.int8,
):
    scale_and_shifted_tensor = tensor / scale + zero_point
    rounded_tensor = torch.round(scale_and_shifted_tensor)

    q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    q_tensor = rounded_tensor.clamp(q_min, q_max).to(dtype)

    return q_tensor

def linear_dequantize_with_scale_and_zero_point(
        tensor : torch.Tensor,
        scale,
        zero_point,
):
    return scale * (tensor.float() - zero_point)

if __name__ == "__main__":
    test_tensor=torch.tensor(
        [[191.6, -13.5, 728.6],
        [92.14, 295.5,  -184],
        [0,     684.6, 245.5]]
        )
    print(f"Original tensor:\n{test_tensor}")
    scale = 3.6
    zero_point = -77

    quantized_tensor = linear_quantize_with_scale_and_zero_point(
        tensor=test_tensor,
        scale=scale,
        zero_point=zero_point,
    )
    print(f"Quantized tensor:\n{quantized_tensor}")
    dequantized_tensor = linear_dequantize_with_scale_and_zero_point(
        tensor=quantized_tensor,
        scale=scale,
        zero_point=zero_point,
    )
    print(f"Dequantized tensor:\n{dequantized_tensor}")

    print(f"Quantization error:\n{test_tensor - dequantized_tensor}")
    print(f"Quantization error square:\n{(dequantized_tensor - test_tensor).square()}")
    print(f"Quantization error square mean:\n{(dequantized_tensor - test_tensor).square().mean()}")

    plot_quantization_errors(test_tensor, quantized_tensor,
                         dequantized_tensor)
    