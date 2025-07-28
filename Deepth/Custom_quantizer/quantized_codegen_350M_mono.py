import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from replace_model_with_quantized_layer import (
    W8A16LinearLayer,
    replace_linear_with_target_and_quantize, 
)
    
if __name__ == "__main__":
    # 1. quantize open source model
    print("---------------------------------------------quantizing open source model 'Salesforce/codegen-350M-mono'---------------------------------------------")
    # TODO：这里可以考虑先将模型下载到本地，然后直接使用本地存储的路径，而不是每次都从网络上下载
    model_id = "Salesforce/codegen-350M-mono"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print(pipe("def hello_world():", max_new_tokens=30, do_sample=False)[0]["generated_text"])
    print(f"model with original layers: \n{model}")

    replace_linear_with_target_and_quantize(model, W8A16LinearLayer, ["lm_head"])
    print(f"model with quantized layers: \n{pipe.model}")
    print(pipe("def hello_world():", max_new_tokens=20, do_sample=False)[0]["generated_text"])