from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch


"""
论文链接 2022：https://arxiv.org/abs/2201.12086
"""


device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
model.to("cuda:1")  # doctest: +IGNORE_RESULT

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "Question: how many cats are there? Answer:"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
inputs['input_ids'] = inputs['input_ids'].to('cuda:1')
inputs['attention_mask'] = inputs['attention_mask'].to('cuda:1')
inputs['pixel_values'] = inputs['pixel_values'].to('cuda:1')
print(inputs)

generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)