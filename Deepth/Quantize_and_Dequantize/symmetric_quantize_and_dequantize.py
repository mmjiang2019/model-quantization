import torch

from model_quantize_and_dequantize.Deepth.Quantize_and_Dequantize.random_quantize_and_dequantize import (
    linear_quantize_with_scale_and_zero_point,
    linear_dequantize_with_scale_and_zero_point
    )

from model_quantize_and_dequantize.Deepth.helper.utils import (
    plot_quantization_errors, 
    quantization_error
    )

# Asymmetric quantization formula:
#   r = s * (q - z)
# where:
# 1. r is the original tensor
# 2. q is the quantized tensor
# 3. s is the scale with same type as r
# 3. z is the zero point with same type as q
# For symmetric quantization, what we need to do is:
# a. make sure the zero point is always 0
# b. make sure the q_max is always the maximum value of dtype
# c. here the r_max should be the maximum value of abs(r)
# Based on the conditions, we can get the symmetric formula:
#   r = s * q
# 
# Then, we can get the scale and zero point calculated directly with logic below:
# a. get scale directly with q_max and r_max of abs(r)
#    s = (r_max) / (q_max)
#    z = 0
# Notes:
# 1. We use round() to round the zero point to the nearest integer
# 2. We use int() to convert the zero point to an integer
def get_quantize_scale_symmetric(tensor: torch.Tensor, dtype: torch.dtype=torch.int8):
    q_max = torch.iinfo(dtype).max
    r_max = tensor.abs().max().item()

    scale = (r_max) / (q_max)

    return scale

def linear_quantize_symmetric(tensor, dtype=torch.int8):
    scale = get_quantize_scale_symmetric(tensor)
    
    quantized_tensor = linear_quantize_with_scale_and_zero_point(
        tensor,
        scale=scale,
        # in symmetric quantization zero point is = 0    
        zero_point=0,
        dtype=dtype)
    
    return quantized_tensor, scale

def linear_dequantize_symmetric(tensor, scale):
    return linear_dequantize_with_scale_and_zero_point(
        tensor,
        scale=scale,
        zero_point=0
    )

if __name__ == '__main__':
    test_tensor=torch.randn((4, 4))

    quantized_tensor, scale = linear_quantize_symmetric(test_tensor)
    print(f"symmetric scale: {scale}")

    dequantized_tensor = linear_dequantize_symmetric(
        quantized_tensor,
        scale
        )
    
    print(f"""Quantization Error : \n{quantization_error(test_tensor, dequantized_tensor)}""")

    plot_quantization_errors("synmmetric_linear_quantization", test_tensor, quantized_tensor, 
                        dequantized_tensor)