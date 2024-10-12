import hydra
from omegaconf import DictConfig, OmegaConf


# python testime/hydra_test.py ...=...,... ...=...,... -m
# 更多关于 yaml 模型配置相关代码见 https://github.com/facebookresearch/segment-anything-2/blob/main/sam2/build_sam.py
def build(config_file: str) -> None:
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]

    # Read config
    config = hydra.compose(config_name=config_file, overrides=hydra_overrides)  # override 一些参数
    OmegaConf.resolve(config)                                                   # 解析配置

    print(config)


if __name__ == "__main__":
    hydra.initialize(config_path="configs")     # 初始化 hydra，一定要记得初始化，否则无法使用 compose
    build("config.yaml")
