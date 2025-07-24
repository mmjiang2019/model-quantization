import torch

from random_quantize_and_dequantize import (
    linear_quantize_with_scale_and_zero_point,
    linear_dequantize_with_scale_and_zero_point
    )

from model_quantize_and_dequantize.Deepth.helper.utils import plot_quantization_errors

# According to the formula:
#   r = s * (q - z)
# where:
# 1. r is the original tensor
# 2. q is the quantized tensor
# 3. s is the scale with same type as r
# 3. z is the zero point with same type as q
# Here we implement the formula with calculated scale and zero point
# The logic is:
# a. get the min and max of the original tensor
# b. get the min and max of the target tensor type
# c. calculate the scale based on the above two values, since:
#    r_max = s * (q_max - z) ---- 1
#    r_min = s * (q_min - z) ---- 2
#   as a result of 1 - 2:
#    r_max - r_min = s * (q_max - q_min)
#   so that:
#    s = (r_max - r_min) / (q_max - q_min)
# d. then we can calculate the zero point from the following equation:
#    r_max = s * (q_max - z)
#    r_min = s * (q_min - z)
#    so that:
#    z = q_max - r_max / s
#    or:
#    z = q_min - r_min / s
# Notes:
# 1. We use round() to round the zero point to the nearest integer
# 2. We use int() to convert the zero point to an integer
def get_quantize_scale_and_zero_point(tensor: torch.Tensor, dtype: torch.dtype=torch.int8):
    q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    r_min, r_max = tensor.min().item(), tensor.max().item()

    scale = (r_max - r_min) / (q_max - q_min)
    zero_point = q_max - r_max / scale

    # extreme cases:
    # clip the zero_point to fall in [quantized_min, quantized_max]
    if zero_point < q_min:
        zero_point = q_min
    elif zero_point > q_max:
        zero_point = q_max

    # round and cast to int
    zero_point = int(round(zero_point))

    return scale, int(zero_point)

if __name__ == '__main__':
    test_tensor=torch.tensor(
    [[191.6, -13.5, 728.6],
     [92.14, 295.5,  -184],
     [0,     684.6, 245.5]]
    )

    new_scale, new_zero_point = get_quantize_scale_and_zero_point(
    test_tensor)
    print(f"new_scale: {new_scale}, new_zero_point: {new_zero_point}")

    quantized_tensor = linear_quantize_with_scale_and_zero_point(
    test_tensor, new_scale, new_zero_point)

    dequantized_tensor = linear_dequantize_with_scale_and_zero_point(quantized_tensor,
                                           new_scale, new_zero_point)
    
    print(f"(dequantized_tensor-test_tensor).square().mean(): {(dequantized_tensor-test_tensor).square().mean()}")

    plot_quantization_errors(test_tensor, quantized_tensor, 
                        dequantized_tensor)