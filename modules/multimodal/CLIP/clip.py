import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIP(nn.Module):
    def __init__(self, *,
                 image_encoder, text_encoder,
                 image_dim, text_dim, embedding_dim,
                 temperature=1.0):
        super(CLIP, self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        # 图像和文本的特征投影
        self.image_projection = nn.Linear(image_dim, embedding_dim, bias=False)
        self.text_projection = nn.Linear(text_dim, embedding_dim, bias=False)

        # 归一化层
        self.l2_norm = lambda x, dim: F.normalize(x, p=2, dim=dim, eps=1e-12, out=None)

        # 相当于起到学习率的作用
        self.temperature = temperature

    def forward(self, image, text):
        # 提取图像和文本的特征
        image_features = self.image_encoder(image)                                      # [N, image_dim]
        text_features = self.text_encoder(text)                                         # [N, text_dim]

        # 联合计算多模态的 embedding
        image_embedding = self.l2_norm(self.image_projection(image_features), dim=1)    # -> [N, embedding_dim]
        text_embedding = self.l2_norm(self.text_projection(text_features), dim=1)       # -> [N, embedding_dim]
        return dict(
            image_features=image_features,
            text_features=text_features,
            image_embedding=image_embedding,
            text_embedding=text_embedding
        )

    def loss(self, image_embedding, text_embedding):
        # 计算 cosine 相似度
        logits = image_embedding @ text_embedding.T * math.exp(self.temperature)    # [N, N]
        print(logits)

        # 计算对称损失
        # 交叉熵损失相当于（以下 input [N, C] 为 logits，target [N] 为 labels）：
        # p = -F.softmax(input, dim=1).log()
        # q = F.one_hot(target, num_classes=C).float()
        # h = torch.sum(p * q) / N
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_image = F.cross_entropy(logits, labels)
        loss_text = F.cross_entropy(logits.T, labels)
        loss = (loss_image + loss_text) / 2
        return loss



