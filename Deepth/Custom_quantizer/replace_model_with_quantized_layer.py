import torch
import torch.nn as nn

from model_quantize_and_dequantize.Deepth.Custom_quantizer.w8a16_linear_layer import W8A16Linear

# Replace all of the torch.nn.Linear layers with the W8A16LinearLayer layer.
# Call quantize on the linear layers using the original weights.
def replace_linear_with_target(
        module, 
        target_class, module_name_to_exclude):
    for name, child in module.named_children():
        if type(child) == nn.Linear and not any([x == name for x in module_name_to_exclude]):
            old_bias = child.bias

            new_module = target_class(
                child.in_features, 
                child.out_features, 
                old_bias is not None, 
                child.weight.dtype)
            
            setattr(module, name, new_module)
            
            if old_bias is not None:
              getattr(module, name).bias = old_bias
            
        else:
            # Recursively call the function for nested modules
            replace_linear_with_target(
                child, 
                target_class,
                module_name_to_exclude)

