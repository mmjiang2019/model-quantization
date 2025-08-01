import torch
import torch.nn as nn
import requests
from PIL import Image

import warnings
# Ignore specific UserWarnings related to max_length in transformers
warnings.filterwarnings("ignore", 
    message=".*Using the model-agnostic default `max_length`.*")

class DummyModel(nn.Module):
  """
  A dummy model that consists of an embedding layer
  with two blocks of a linear layer followed by a layer
  norm layer.
  """
  def __init__(self, input_feature: int, output_feature: int):
    # Initialize the model
    # TODO: remove this assertion in the future
    assert input_feature == output_feature, "input_feature and output_feature must be the same"
    super().__init__()

    torch.manual_seed(123)

    self.token_embedding = nn.Embedding(input_feature, output_feature)

    # Block 1
    self.linear_1 = nn.Linear(input_feature, output_feature)
    self.layernorm_1 = nn.LayerNorm(output_feature)

    # Block 2
    self.linear_2 = nn.Linear(input_feature, output_feature)
    self.layernorm_2 = nn.LayerNorm(output_feature)

    self.head = nn.Linear(input_feature, output_feature)

  def forward(self, x):
    hidden_states = self.token_embedding(x)

    # Block 1
    hidden_states = self.linear_1(hidden_states)
    hidden_states = self.layernorm_1(hidden_states)

    # Block 2
    hidden_states = self.linear_2(hidden_states)
    hidden_states = self.layernorm_2(hidden_states)

    logits = self.head(hidden_states)
    return logits


def get_generation(model, processor, image, dtype):
  inputs = processor(image, return_tensors="pt").to(dtype)
  out = model.generate(**inputs)
  return processor.decode(out[0], skip_special_tokens=True)


def load_image(img_url):
    image = Image.open(requests.get(
        img_url, stream=True).raw).convert('RGB')

    return image