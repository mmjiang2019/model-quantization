import os
# 注意os.environ得在import huggingface库相关语句之前执行
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import requests
import torch

from PIL import Image
from IPython.display import display

from transformers import AutoProcessor, BlipForConditionalGeneration

from helper.utils import get_generation, load_image

model_name = "Salesforce/blip-image-captioning-base"
processor = AutoProcessor.from_pretrained(model_name)

print(f"model memory")
model = BlipForConditionalGeneration.from_pretrained(model_name)
fp32_mem_footprint = model.get_memory_footprint()
print(f"Footprint of the fp32 model in bytes: {fp32_mem_footprint}, in MBs: {fp32_mem_footprint/1e+6}")

model_bf16 = BlipForConditionalGeneration.from_pretrained(model_name,torch_dtype=torch.bfloat16)
bf16_mem_footprint = model_bf16.get_memory_footprint()
print(f"Footprint of the bfp16 model in bytes: {bf16_mem_footprint}, in MBs: {bf16_mem_footprint/1e+6}")

# Get the relative difference
relative_diff = bf16_mem_footprint / fp32_mem_footprint
print(f"Relative diff of bf16 and fp32 in bytes: {relative_diff}")

print(f"model inference performance")
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
image = load_image(img_url)
display(image.resize((500, 350)))

results_fp32 = get_generation(model, 
                              processor, 
                              image, 
                              torch.float32)
results_bf16 = get_generation(model_bf16, 
                              processor, 
                              image, 
                              torch.bfloat16)
print(f"fp32 Model Results:\n{results_fp32}\n\nbf16 Model Results:\n{results_bf16}")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "A picture of"

inputs = processor(text=text, images=image, return_tensors="pt")

outputs = model(**inputs)
print(f"outputs: {outputs}")