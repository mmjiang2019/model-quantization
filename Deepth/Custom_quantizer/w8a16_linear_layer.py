import torch
import torch.nn as nn
import torch.nn.functional as F

class W8A16LinearLayer(nn.Module):
    '''
    W8A16LinearLayer
                    # 8-bit  # 16-bit         # optional
    * w8_a16_forward -> weights, input,   scales, bias=None
    Cast the 8-bit weights to the same data type as the input, "casted weights",
    keeping the "casted weights" in the same range as before, [-128, 127]
    Next,
    ((𝑖𝑛𝑝𝑢𝑡𝑠⋅``casted weights'')∗𝑠𝑐𝑎𝑙𝑒)+𝑏𝑖𝑎𝑠
    '''
    def __init__(self, in_features: int, out_features: int, bias: bool=True, dtype: torch.Tensor = torch.float32):
        super().__init__()
        
        self.register_buffer(
            "int8_weights",
            torch.randint(
                -128, 127,
                (out_features, in_features), 
                dtype=torch.int8,
                )
        )

        self.register_buffer(
            "scales",
            torch.randn((out_features), dtype=dtype)
            )
        
        if bias:
            self.register_buffer(
                "bias",
                torch.randn((1, out_features), dtype=dtype),
                )
        else:
            self.bias = None

    def quantize(self, weights: torch.Tensor):
        w_fp32 = weights.clone().to(torch.float32)
        scales = w_fp32.abs().max(dim=-1).values / 127
        scales = scales.to(weights.dtype)

        int8_weights = torch.round(weights/scales.unsqueeze(1)).to(torch.int8)

        self.int8_weights = int8_weights
        self.scales = scales
    
    def w8_a16_forward(self, weight: torch.Tensor, input: torch.Tensor, scales: torch.Tensor, bias: torch.Tensor = None):
        casted_weights = weight.to(input.dtype)
        output = F.linear(input, casted_weights) * scales
        
        if bias is not None:
            output = output + bias
        
        return output
    
    def forward(self, input: torch.Tensor):
        return self.w8_a16_forward(self.int8_weights, input, self.scales, self.bias)


if __name__ == '__main__':
    module = W8A16LinearLayer(4, 8)
    print("Weights before:\n" , module.int8_weights)
    print(f"weiths shape: \n{module.int8_weights.shape}")
    print(f"scales shape: \n{module.scales.shape}")

    random_matrix = torch.randn((4, 8), dtype=torch.bfloat16)
    module.quantize(random_matrix)
    print("Weights After:\n" , module.int8_weights)
    print(f"weiths shape: \n{module.int8_weights.shape}")
    print(f"scales shape: \n{module.scales.shape}")