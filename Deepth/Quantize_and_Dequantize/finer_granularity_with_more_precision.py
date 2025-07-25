import torch

from model_quantize_and_dequantize.Deepth.helper.utils import (
    plot_quantization_errors, 
    quantization_error
    )

from model_quantize_and_dequantize.Deepth.Quantize_and_Dequantize.random_quantize_and_dequantize import (
    linear_quantize_with_scale_and_zero_point
    )

from model_quantize_and_dequantize.Deepth.Quantize_and_Dequantize.symmetric_quantize_and_dequantize import (
    get_quantize_scale_symmetric,
    linear_quantize_symmetric, 
    linear_dequantize_symmetric
    )

# here we discuss the linear quantization base on different granularities:
# 1. per tensor
# 2. per channel
# 3. per group
# Notes:
# 1. In order to make the quantization understood easily, here we only discuss symmetric quantization
# 2. Since we've already discussed the symmetric quantization per tensor before, here we only discuss the symmetric dequantization per channel/group
# 3. for simplicity, we only discuss 2D symmetric quantization per channel/group

def linear_quantize_symmetric_per_channel(r_tensor: torch.Tensor, dim: int, dtype: torch.dtype=torch.int8):
    output_dim = r_tensor.shape[dim]
    # store the scales
    scale = torch.zeros(output_dim)

    for i in range(output_dim):
        sub_tensor = r_tensor.select(dim, i)
        scale[i] = get_quantize_scale_symmetric(sub_tensor, dtype)

    # reshape the scale
    # here we keep all other dimension as 1, and use the dimension 
    # inference to infer the number of target dimension for stored scale
    # actually, it's the same result of scale.unsqueeze(-1)
    scale_shape = [1] * r_tensor.dim()
    scale_shape[dim] = -1 
    scale = scale.view(scale_shape)

    quantized_tensor = linear_quantize_with_scale_and_zero_point(r_tensor,scale, 0, dtype)

    return quantized_tensor, scale

# here we only discuss the symmetric quantization per group on dimension 0 with number of dimension 2
def linear_quantize_symmetric_per_group(r_tensor: torch.Tensor, group_size: int, dtype: torch.dtype=torch.int8):
    dim = r_tensor.dim()
    assert dim == 2, "only support 2D tensor"

    # store the original shape
    shape = r_tensor.shape
    assert shape[1] % group_size == 0, f"the dimension 1 should be divisible by group size"

    # change the tensor view, convert dimension 1 to group_size
    r_tensor = r_tensor.view(-1, group_size)
    
    quantized_tensor, scale = linear_quantize_symmetric_per_channel(r_tensor, dim=0, dtype=dtype)

    # restore the tensor view
    quantized_tensor = quantized_tensor.view(shape)

    return quantized_tensor, scale

def linear_dequantization_symmetric_per_group(quantized_tensor: torch.Tensor, scale, group_size: int):
    quantized_tensor_shape = quantized_tensor.shape

    quantized_tensor = quantized_tensor.view(-1, group_size)

    dequantized_tensor = linear_dequantize_symmetric(quantized_tensor, scale)

    dequantized_tensor = dequantized_tensor.view(quantized_tensor_shape)

    return dequantized_tensor

if __name__ == '__main__':
    # 1. symmetric quantization per tensor
    test_tensor=torch.tensor(
        [
        [191.6, -13.5, 728.6],
        [92.14, 295.5,  -184],
        [0,     684.6, 245.5]
        ])
    
    quantized_tensor, scale =linear_quantize_symmetric(test_tensor)

    print("per tensor symmetric quantization result: ")
    print(quantized_tensor)

    dequantized_tensor = linear_dequantize_symmetric(quantized_tensor, scale)

    print("per tensor symmetric dequantization result: ")
    print(dequantized_tensor)

    print("per tensor symmetric quantization error: ")
    print(quantization_error(test_tensor, dequantized_tensor))
   
    plot_quantization_errors("per tensor", test_tensor, quantized_tensor, dequantized_tensor, show=False)

    # 2. symmetric quantization per channel
    for i in range(test_tensor.dim()):
        print("quantize per channel {}".format(i))
        quantized_tensor, scale = linear_quantize_symmetric_per_channel(test_tensor, dim=i)

        print("per channel symmetric quantization result: ")
        print(quantized_tensor)

        dequantized_tensor = linear_dequantize_symmetric(quantized_tensor, scale)

        print("per channel symmetric dequantization result: ")
        print(dequantized_tensor)

        print("per channel symmetric quantization error: ")
        print(quantization_error(test_tensor, dequantized_tensor))
        
        # show = True if i == test_tensor.dim()-1 else False
        plot_quantization_errors(f"per channel {i}", test_tensor, quantized_tensor, dequantized_tensor, show=False)


    # 3. symmetric quantization per group
    test_tensor = torch.rand((6, 6))
    group_size = 3
    quantized_tensor, scale = linear_quantize_symmetric_per_group(test_tensor, group_size)

    dequantized_tensor = linear_dequantization_symmetric_per_group(quantized_tensor, scale,group_size)
    
    print("per channel symmetric quantization error: ")
    print(quantization_error(test_tensor, dequantized_tensor))
    plot_quantization_errors(f"per group {3}", test_tensor, quantized_tensor, dequantized_tensor, show=True)
