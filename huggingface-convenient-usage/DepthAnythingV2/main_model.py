from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os


# load model: 加载模型和模型参数
model_name = "depth-anything/Depth-Anything-V2-Small-hf"
image_processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForDepthEstimation.from_pretrain(model_name)

# load image: 加载图像
image_name = "example.jpg"
image = Image.open(f"./figures/img/{image_name}")

# pre-processing: 图像预处理
inputs = image_processor(images=image, return_tensors="pt")

# inference: 模型推理
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth: torch.Tensor = outputs.predicted_depth

prediction = F.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size,
    mode="bicubic",
    align_corners=False
)

# post-processing: 后处理
output = prediction.squeeze().cpu().numpy()
formatted = (output * 255 / np.max(output)).astype(np.uint8)
depth = Image.fromarray(formatted)

# save image: 保存图像
os.makedirs("./figures/depth", exist_ok=True)
depth.save(f"./figures/depth/{image_name}")