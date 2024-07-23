import os

from utils.log.info import print_


model_dir = "./models/"
local_cache_dir = ".cache/"


def load_model(model_base, version: str = "openai/clip-vit-base-patch32"):
    """
    从本地加载模型，如果本地没有模型，则从 huggingface 下载模型并保存到本地
    """
    model_path = os.path.join(model_dir, version)
    os.makedirs(model_path, exist_ok=True)
    model = None
    try:
        # 尝试从本地加载模型
        model = model_base.from_pretrained(model_path)
        print_(f"Model loaded from {model_path}")
    finally:
        # 如果本地没有模型，则从 huggingface 下载模型并保存到本地
        if model is None:
            model = model_base.from_pretrained(version)
            model.save_pretrained(model_path)
            print_(f"Model loaded from hugggingface: {version}")

    return model
