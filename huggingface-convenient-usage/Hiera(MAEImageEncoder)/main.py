import torch
from PIL import Image
import os
from torchvision import transforms


"""
必要的文件层级
└── figures
    └── example.jpg
"""


def load_model(*, model_repository, model_dir=None, model_name, pretrained, checkpoint, **kwargs):
    """
    如果本地有保存模型.safeTensor文件就读取，否则加载模型并保存
    Args:
        model_repository (str): 模型仓库
        model_dir (str): 模型保存路文件夹
        model_name (str): 模型名称
        pretrained (bool): 是否加载预训练模型
        checkpoint (str): 加载预训练模型的检查点
        **kwargs: 其他参数
    """
    model_path: str = os.path.join(model_dir, model_name)
    if os.path.exists(model_path):
        model = torch.hub.load(
            model_path,
            model=model_name, pretrained=pretrained, checkpoint=checkpoint, **kwargs)
    else:
        torch.hub.set_dir(model_dir)
        model = torch.hub.load(
            model_repository,
            model=model_name, pretrained=pretrained, checkpoint=checkpoint, **kwargs)

    return model


# load model: 加载模型和模型参数，以下是如果本地有保存模型文件就读取，否则加载模型并保存
# 在 README.md 文件中可以找到 Pre-trained Models 的 size
model_name = "hiera_tiny_224"   # hiera_tiny_224, hiera_small_224, hiera_base_224, hiera_large_224, hiera_huge_224
model = load_model(
    model_repository="facebookresearch/hiera",
    model_dir="./models",
    model_name=model_name,
    pretrained=True,
    checkpoint="mae_in1k"       # 如果有微调，则在后面加上 _ft_in1k
)

# load image: 加载图像
image_name = "example.jpg"
image = Image.open(os.path.join("./figures/", image_name))

# preprocess: 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image = transform(image)
image = image.unsqueeze(0)

# inference: 模型推理
depth = model(image)
print(depth.shape)
print(depth)
