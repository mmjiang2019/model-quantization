import torch

if __name__ == '__main__':
    # Information of `8-bit unsigned integer`
    print(f"torch.iinfo(torch.uint8): {torch.iinfo(torch.uint8)}")
    # Information of `8-bit (signed) integer`
    print(f"torch.iinfo(torch.int8): {torch.iinfo(torch.int8)}")

    ### Information of `64-bit (signed) integer`
    print(f"torch.iinfo(torch.int64): {torch.iinfo(torch.int64)}")

    ### Information of `32-bit (signed) integer`
    print(f"torch.iinfo(torch.int32): {torch.iinfo(torch.int32)}")

    ### Information of `16-bit (signed) integer`
    print(f"torch.iinfo(torch.int16): {torch.iinfo(torch.int16)}")

    # by default, python stores float data in fp64
    value = 1/3
    print(f"value 1/3: {format(value, '.60f')}")

    # 64-bit floating point
    tensor_fp64 = torch.tensor(value, dtype = torch.float64)
    tensor_fp32 = torch.tensor(value, dtype = torch.float32)
    tensor_fp16 = torch.tensor(value, dtype = torch.float16)
    tensor_bf16 = torch.tensor(value, dtype = torch.bfloat16)
    print(f"fp64 tensor: {format(tensor_fp64.item(), '.60f')}")
    print(f"fp32 tensor: {format(tensor_fp32.item(), '.60f')}")
    print(f"fp16 tensor: {format(tensor_fp16.item(), '.60f')}")
    print(f"bf16 tensor: {format(tensor_bf16.item(), '.60f')}")

    # Information of `16-bit brain floating point`
    print(f"torch.finfo(torch.bfloat16): {torch.finfo(torch.bfloat16)}")

    # Information of `32-bit floating point`
    print(f"torch.finfo(torch.float32): {torch.finfo(torch.float32)}")

    ### Information of `16-bit floating point`
    print(f"torch.finfo(torch.float16): {torch.finfo(torch.float16)}")