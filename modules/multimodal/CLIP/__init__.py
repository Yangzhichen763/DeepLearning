from typing import List

import torch
import torch.nn as nn
import transformers
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer, CLIPTextModel
from utils.transformers import load_model

import PIL.Image


"""
如果出现 HTTPSConnectionPool(host='huggingface.co', port=443)：
  - 降级版本，魔法上网，执行如下指令
    pip install requests==2.27.1
    pip install urllib3==1.25.11

CLIP: Contrastive Language-Image Pre-training
论文链接 2021：https://arxiv.org/abs/2103.00020
"""


class CLIP(nn.Module):
    def __init__(self, version: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.model = load_model(CLIPModel, version)
        self.processor = load_model(CLIPProcessor, version)

    def __call__(self, texts: List[str], images, device):
        """
        Returns:
            outputs: outputs.logits_per_image.softmax(dim=-1) 获取每个图片对应的文本置信度 (batch_size, num_texts)
                     outputs.logits_per_text.softmax(dim=-2) 获取每个文本对应的图片置信度 (num_texts, batch_size)
                     outputs.image_embeds 图片的向量化表示 (batch_size, 512)
                     outputs.text_embeds 文本的向量化表示 (num_texts, 512)
        """
        self.model.to(device)
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}   # 将 inputs 转移到 device 上
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs


class CLIPTextEmbedder(nn.Module):
    """
    ## CLIP Text Embedder
    """

    def __init__(self, version: str = "openai/clip-vit-large-patch14", device="cuda:0", max_length: int = 77):
        """
        :param version: is the model version
        :param device: is the device
        :param max_length: is the max length of the tokenized prompt
        """
        super().__init__()
        # Load the tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        # Load the CLIP transformer
        self.transformer = CLIPTextModel.from_pretrained(version).eval()

        self.device = device
        self.max_length = max_length

    def forward(self, prompts: List[str]):
        """
        :param prompts: are the list of prompts to embed
        """
        # Tokenize the prompts
        batch_encoding = self.tokenizer(prompts, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        # Get token ids
        tokens = batch_encoding["input_ids"].to(self.device)
        # Get CLIP embeddings
        return self.transformer(input_ids=tokens).last_hidden_state


if __name__ == '__main__':
    _model = CLIP('openai/clip-vit-base-patch32')
    _text = ["a photo of a cat", "a photo of a dog"]
    _image = PIL.Image.open("pictures/cat.jpg")
    _outputs = _model(_text, _image, device='cuda:0')
    print(_outputs)
