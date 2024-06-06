import concurrent
import os
from tqdm import tqdm

import numpy as np
import torch
import PIL.Image as Image
from pycocotools.coco import COCO
from utils import save, general
import time


"""
由于 RoboFlow 中导出的数据集标签是几何图形，因此无法直接用于训练，需要将其转化为 mask 图像。
这里使用 pycocotools 库将 coco 格式的 json 文件转化为 mask 图像。
"""


def save_image(image, save_path):
    image.save(save_path)

def json_to_mask(json_path, save_dir):
    """
    将 coco 格式的 json 文件转化为 mask 图像
    Args:
        json_path: coco 格式的 json 文件路径
        save_dir: 保存图像和 mask 的目录
    """
    # 读取json
    coco = COCO(json_path)          # 读取 coco 格式的 json 文件
    images_ids = coco.getImgIds()   # 获取图像 id 列表

    general.buffer_dataloader(None)
    pbar = tqdm(
            iterable=images_ids,
            desc=f"Phase {_dir}",
            unit='image')
    # 遍历每张图像
    with pbar:
        for img_id in pbar:
            # image_info = { id, license, file_name, height, width, date_captured }
            image_info = coco.loadImgs(img_id)[0]
            ann_ids = coco.getAnnIds(imgIds=img_id)     # 获取对应标注的序号
            anns = coco.loadAnns(ann_ids)               # 获取标注信息

            shape = (int(image_info['height']), int(image_info['width']))
            mask = torch.zeros(shape, device="cuda").to(torch.uint8)
            for ann in anns:
                _mask = coco.annToMask(ann)                                                 # 解码
                mask.bitwise_or_(torch.from_numpy(_mask).to(torch.uint8).to(mask.device))   # 合并
            mask[mask == 1] = 255  # 调整灰度值便于显示

            os.makedirs(save_dir, exist_ok=True)
            # 加载并保存原图像到目标位置
            # 加载原图像
            image_load_path = os.path.join(os.path.dirname(json_path), image_info['file_name'])
            image = Image.open(image_load_path)
            # 保存原图像
            image_file_name = image_info['file_name']
            image_save_path = os.path.join(save_dir, "images", image_file_name)
            os.makedirs(os.path.dirname(image_save_path), exist_ok=True)
            image.save(image_save_path)

            # 保存 mask 图像
            i_dot = image_file_name.rindex('.')
            mask_file_name = image_file_name[:i_dot] + "_mask" + image_file_name[i_dot:]
            mask_save_path = os.path.join(save_dir, "masks", mask_file_name)
            save.to_image(mask, save_path=mask_save_path)


if __name__ == '__main__':
    for _dir in ["train", "valid", "test"]:
        print(f"\n{_dir}:")
        json_to_mask(
            json_path=f"E:/Developments/PythonLearning/_datasets/RebarSegmentation.v1i.coco-segmentation/{_dir}/_annotations.coco.json",
            save_dir=f"../datas/RebarSegmentation/{_dir}")
