import torch
import torch.nn as nn

from model_quantize_and_dequantize.Deepth.Custom_quantizer.w8a16_linear_layer import W8A16LinearLayer

# Replace all of the torch.nn.Linear layers with the W8A16LinearLayer layer.
# Call quantize on the linear layers using the original weights.
def replace_linear_with_target_and_quantize(
        module, 
        target_class, module_name_to_exclude):
    for name, child in module.named_children():
        # 这里 isinstance(child, nn.Linear) 和 type(child) == nn.Linear 都是用于判断对象的类型，
        if type(child) == nn.Linear and not any([x == name for x in module_name_to_exclude]):
            old_bias = child.bias
            old_weight = child.weight

            new_module = target_class(
                child.in_features, 
                child.out_features, 
                old_bias is not None, 
                child.weight.dtype)
            
            # replace the old linear layer with the new one
            setattr(module, name, new_module)

            # perform quantization
            getattr(module, name).quantize(old_weight)
            
            if old_bias is not None:
              getattr(module, name).bias = old_bias
            
        else:
            # Recursively call the function for nested modules
            replace_linear_with_target_and_quantize(
                child, 
                target_class,
                module_name_to_exclude)

class DummyModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embedding = torch.nn.Embedding(10, 10)
        # try with bias
        self.linear1 = torch.nn.Linear(10, 10, bias=True)
        # try without bias
        self.linear2 = torch.nn.Linear(10, 10, bias=False)
        # lm prediction head
        self.lm_head = torch.nn.Linear(10, 10, bias=False)

if __name__ == "__main__":
    # replace layers with quantized layers
    print("---------------------------------------------replace layers with quantized layers on custom model---------------------------------------------")
    model_1 = DummyModel()
    model_2 = DummyModel()

    print(f"model 1 with original layers: \n{model_1}")
    replace_linear_with_target_and_quantize(model_1, W8A16LinearLayer, ["lm_head"])
    print(f"model 1 with quantized layers: \n{model_1}")

    print(f"model 2 with original layers: \n{model_2}")
    replace_linear_with_target_and_quantize(model_2, W8A16LinearLayer, [])
    print(f"model 2 with quantized layers: \n{model_2}")