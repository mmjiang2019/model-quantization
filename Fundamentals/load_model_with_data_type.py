from copy import deepcopy
import torch
import torch.nn as nn

from model_quantize_and_dequantize.Fundamentals.helper.utils import DummyModel

def print_param_dtype(model: nn.Module):
    for name, param in model.named_parameters():
        print(f"{name} is loaded in {param.dtype}")


if __name__ == "__main__":
    model = DummyModel(2, 2)

    print(f"\n\nmodel_fp32:\n{model}\n")
    dummy_input = torch.LongTensor([[1, 0], [0, 1]])
    # inference using float32 model
    fp32_outpuy = model(dummy_input)
    print(f"fp32_outpuy: \n{fp32_outpuy}\n")

    # float 16
    model_fp16 = DummyModel(2, 2).half()
    print(f"\n\nmodel casting to float16\nmodel_fp16:\n{model_fp16}\n")

    # inference using float16 model
    # Note:
    # In old version, the following code will raise an error as below:
    # RuntimeError :  "addmm_impl_cpu_" not implemented for 'Half' 
    try:
        fp16_outpuy = model_fp16(dummy_input)
        print(f"fp16_outpuy: \n{fp16_outpuy}\n")
    except Exception as error:
        print("\033[91m", type(error).__name__, ": ", error, "\033[0m")

    # brain float 16
    model_bfp16 = deepcopy(model).to(torch.bfloat16)
    print(f"\n\nmodel casting to brain float16\nmodel_bfp16:\n{model_bfp16}\n")

    # inference using brain float16 model
    try:
        bfp16_outpuy = model_bfp16(dummy_input)
        print(f"bfp16_outpuy: \n{bfp16_outpuy}\n")
    except Exception as error:
        print("\033[91m", type(error).__name__, ": ", error, "\033[0m")

    print(f"compare the difference of bfloat16 and float32")
    mean_diff = torch.abs(bfp16_outpuy - fp32_outpuy).mean().item()
    max_diff = torch.abs(bfp16_outpuy - fp32_outpuy).max().item()
    print(f"Mean diff: {mean_diff} | Max diff: {max_diff}")

    print(f"compare the difference of bfloat16 and float16")
    mean_diff = torch.abs(bfp16_outpuy - fp16_outpuy).mean().item()
    max_diff = torch.abs(bfp16_outpuy - fp16_outpuy).max().item()
    print(f"Mean diff: {mean_diff} | Max diff: {max_diff}")