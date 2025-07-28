import torch

from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import requests

from replace_model_with_quantized_layer import (
    W8A16LinearLayer,
    replace_linear_with_target_and_quantize, 
)

from model_quantize_and_dequantize.Deepth.helper.utils import plot_results
    
if __name__ == "__main__":
    # quantize open source model
    print("---------------------------------------------quantizing open source model 'facebook/detr-resnet-50'---------------------------------------------")
    # TODO：这里可以考虑先将模型下载到本地，然后直接使用本地存储的路径，而不是每次都从网络上下载
    # you can specify the revision tag if you don't want the timm dependency
    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50", revision="no_timm")

    previous_memory_footprint = model.get_memory_footprint()
    print(f"Footprint of the model before quantization in MBs: \n {previous_memory_footprint/1e+6}")

    img_path = "Deepth/Custom_quantizer/dinner_with_friends.png"
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    plot_results("DETR before quantization", model, image, results, show=False)

    replace_linear_with_target_and_quantize(model, 
                                        W8A16LinearLayer, 
               ["0", "1", "2", "class_labels_classifier"])

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    new_memory_footprint = model.get_memory_footprint()
    print(f"Footprint of the model after quantization in MBs: \n {new_memory_footprint/1e+6}")

    ### Memory saved
    print("Memory saved in MBs: ", 
        (previous_memory_footprint - new_memory_footprint)/1e+6)
    plot_results("DETR after quantization", model, image, results, show=True)