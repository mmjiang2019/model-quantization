import torch

# This file describes how packing and unpacking weights into int8 tensors works
# 
# ========================================================================
# Packing Logic
# ========================================================================
# The packing is done by iterating over the uint8 tensor and packing the bits
# Example Tensor: [1, 0, 3, 2]
    # 1 0 3 2 - 01 00 11 10

    # Starting point of packed int8 Tensor
    # [0000 0000]

    ##### First Iteration Start:
    # packed int8 Tensor State: [0000 0000]
    # 1 = 0000 0001
    # 0000 0001
    # No left shifts in the First Iteration
    # After bit-wise OR operation between 0000 0000 and 0000 0001:
    # packed int8 Tensor State: 0000 0001
    ##### First Iteration End

    ##### Second Iteration Start:
    # packed int8 Tensor State: [0000 0001]
    # 0 = 0000 0000
    # 0000 0000
    # 2 left shifts:
    # [0000 0000] (1 shift)-> 0000 0000 (2 shift)-> 0000 0000
    # After bit-wise OR operation between 0000 0001 and 0000 0000:
    # packed int8 Tensor State: 0000 0001
    ##### Second Iteration End

    ##### Third Iteration Start:
    # packed int8 Tensor State: [0000 0001]
    # 3 = 0000 0011
    # 0000 0011
    # 4 left shifts:
    # [0000 0011] (1 shift)-> 0000 0110 (2 shift)-> 0000 1100
    # 0000 1100 (3 shift)-> 0001 1000 (4 shift)-> 0011 0000
    # After bit-wise OR operation between 0000 0001 and 0011 0000:
    # packed int8 Tensor State: 0011 0001
    ##### Third Iteration End

    ##### Fourth Iteration Start:
    # packed int8 Tensor State: [0011 0001]
    # 2 = 0000 0010
    # 0000 0010
    # 6 left shifts:
    # [0000 0010] (1 shift)-> 0000 0100 (2 shift)-> 0000 1000
    # 0000 1000 (3 shift)-> 0001 0000 (4 shift)-> 0010 0000
    # 0010 0000 (5 shift)-> 0100 0000 (6 shift)-> 1000 0000
    # After bit-wise OR operation between 0011 0001 and 1000 0000:
    # packed int8 Tensor State: 1011 0001
    ##### Fourth Iteration End

    # Final packed int8 Tensor State: [1011 0001]

# ========================================================================
# UnPacking Logic
# ========================================================================
# The logic below describes how unpacking int8 tensors works
# Example Tensor: [10110001]
    # Which was Originally: 1 0 3 2 - 01 00 11 10

    # Starting point of unpacked Tensor
    # [00000000 00000000 00000000 00000000]

    ##### First Iteration Start:
    # packed int8 Tensor: [10110001]
    # You want to extract 01 from [101100 01]
    # No right shifts in the First Iteration
    # After bit-wise OR operation between 00000000 and 10110001:
    # [10110001 00000000 00000000 00000000]
    # unpacked Tensor state: [10110001 00000000 00000000 00000000]
    ##### First Iteration End

    ##### Second Iteration Start:
    # packed int8 Tensor: [10110001]
    # You want to extract 00 from [1011 00 01]
    # 2 right shifts:
    # [10110001] (1 shift)-> 01011000 (2 shift)-> 00101100
    # After bit-wise OR operation between 00000000 and 00101100:
    # [10110001 00101100 00000000 00000000]
    # unpacked Tensor state: [10110001 00101100 00000000 00000000]
    ##### Second Iteration End

    ##### Third Iteration Start:
    # packed int8 Tensor: [10110001]
    # You want to extract 11 from [10 11 0001]
    # 4 right shifts:
    # [10110001] (1 shift)-> 01011000 (2 shift)-> 00101100
    # 00101100 (3 shift)-> 00010110 (4 shift)-> 00001011
    # After bit-wise OR operation between 00000000 and 00001011:
    # [10110001 00101100 00001011 00000000]
    # unpacked Tensor state: [10110001 00101100 00001011 00000000]
    ##### Third Iteration End

    ##### Fourth Iteration Start:
    # packed int8 Tensor: [10110001]
    # You want to extract 10 from [10 110001]
    # 6 right shifts:
    # [10110001] (1 shift)-> 01011000 (2 shift)-> 00101100
    # 00101100 (3 shift)-> 00010110 (4 shift)-> 00001011
    # 00001011 (5 shift)-> 00000101 (6 shift)-> 00000010
    # After bit-wise OR operation between 00000000 and 00000010:
    # [10110001 00101100 00001011 00000010]
    # unpacked Tensor state: [10110001 00101100 00001011 00000010]
    ##### Fourth Iteration End

    # Last step: Perform masking (bit-wise AND operation)
    # Mask: 00000011
    # Bit-wise AND operation between 
    # unpacked Tensor and 00000011
    # [10110001 00101100 00001011 00000010] <- unpacked tensor
    # [00000011 00000011 00000011 00000011] <- Mask
    # [00000001 00000000 00000011 00000010] <- Result

    # Final
    # unpacked Tensor state: [00000001 00000000 00000011 00000010]


def pack_weights(unpacked_tensor, bits):
    if unpacked_tensor.shape[0] * bits % 8 != 0:
        raise ValueError(f"The input shape needs to be a mutiple of {8 / bits} - got {uint8tensor.shape[0]}")
    
    num_values = unpacked_tensor.shape[0] * bits // 8
    num_steps = 8 // bits
    
    # 1 0 3 2 - 01 00 11 10
    # [0000 0000] -> 0000 0001
    # 0000 0001
    # 0000 0000 - 0000 0000
    # 0000 0011 - 0011 0000 - 0011 0001
    # 1011 0001

    unpacked_idx = 0
    packed_tensor = torch.zeros((num_values), dtype=torch.uint8)

    for i in range(num_values):
        for j in range(num_steps):
            packed_tensor[i] |= unpacked_tensor[unpacked_idx] << (bits * j)
            unpacked_idx += 1

    return packed_tensor

def unpack_weights(packed_tensor, bits):
    num_values = packed_tensor.shape[0] * 8 // bits
    num_steps = 8 // bits

    unpack_weights = torch.zeros((num_values), dtype=torch.uint8)

    # 1 0 3 2 - 01 00 11 10
    # [00000000 00000000 00000000 00000000]
    # [10110001 00101100 00001011 00000010]
    # [00000001 00000000 00000011 00000010]
    # 10110001
    # 00000011
    # 00000001
    # 1: [10110001]
    # 2: [00101100]
    # 3: [00001011]
    unpack_index = 0

    for i in range(packed_tensor.shape[0]):
        for j in range(num_steps):
            unpack_weights[unpack_index] |= packed_tensor[i] >> (bits * j)
            unpack_index += 1

    # Masking all the bits with mask: 2 ** bits - 1
    unpack_weights &= ((1 << bits) - 1)

    return unpack_weights

if __name__ == '__main__':
    unpacked_tensor = torch.tensor([1, 0, 3, 2], dtype=torch.uint8)
    packed_tensor = pack_weights(unpacked_tensor, 2)
    unpacked_tensor_res = unpack_weights(packed_tensor, 2)
    print(f"Unpacked tensor:\n{unpacked_tensor}\n\nPacked tensor:\n{packed_tensor}\n\nUnpacked result:\n{unpacked_tensor_res}")

    unpacked_tensor = torch.tensor([1, 0, 3, 2, 3, 3, 3, 3], dtype=torch.uint8)
    packed_tensor = pack_weights(unpacked_tensor, 2)
    unpacked_tensor_res = unpack_weights(packed_tensor, 2)
    print(f"Unpacked tensor:\n{unpacked_tensor}\n\nPacked tensor:\n{packed_tensor}\n\nUnpacked result:\n{unpacked_tensor_res}")