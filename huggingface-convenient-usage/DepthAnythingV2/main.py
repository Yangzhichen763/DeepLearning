from transformers import pipeline
from PIL import Image
import os


"""
必要的文件层级
├── figures
│   └── img
│       └── example.jpg
└── models
    └── Depth-Anything-V2-Small-hf
"""


def load_pipe(*, task, model, model_path):
    """
    如果本地有保存模型.safeTensor文件就读取，否则加载模型并保存
    Args:
        task: 任务类型，如 depth-estimation
        model: 选择的模型，如 depth-anything/Depth-Anything-V2-Small-hf
        model_path: 模型保存路径
    """
    if os.path.exists(model_path):
        pipe = pipeline(task=task, model=model_path)
    else:
        pipe = pipeline(task=task, model=model)
        pipe.save_pretrained(model_path)

    return pipe


# load pipe: 加载模型和模型参数，以下是如果本地有保存模型.safeTensor文件就读取，否则加载模型并保存
# 在 README.md 文件中可以找到 Pre-trained Models 的 size
pipe = load_pipe(
    task="depth-estimation",                            # 任务类型，选择深度估计就用这个
    model="depth-anything/Depth-Anything-V2-Small-hf",  # 选择的模型，选择 DepthAnythingV2 模型，就用这个，其中 Small 代表模型大小
    model_path="./models/Depth-Anything-V2-Small-hf"    # 模型本地保存路径
)

# load image: 加载图像
image_name = r'example.jpg'
image = Image.open(rf'./figures/img/{image_name}')

# inference: 模型推理
depth = pipe(image)["depth"]

# save image: 保存图像
os.makedirs(r"./figures/depth", exist_ok=True)
depth.save(rf"./figures/depth/{image_name}")