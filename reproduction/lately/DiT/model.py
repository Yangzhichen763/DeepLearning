from typing import Optional

import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import PatchEmbed, Mlp, Attention
from modules.embedding.timestep import get_timestep_embedding
from modules.embedding import get_1d_sinusoidal_positional_embedding


# 论文链接：https://arxiv.org/abs/2212.09748
# 代码参考：https://github.com/facebookresearch/DiT/blob/main/models.py


# 对 Diffusion 过程的 timestep 进行 embedding
class TimestepEmbedder(nn.Module):
    """
    将 time step 嵌入到向量表示中
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True)
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, x):
        frequency_embedding = get_timestep_embedding(x, self.frequency_embedding_size)
        timestep_embedding = self.mlp(frequency_embedding)
        return timestep_embedding


# 对生成方向、标签进行 embedding
class LabelEmbedder(nn.Module):
    """
    将标签嵌入到向量表示中，同时也可以通过 dropout 处理 classifier-free 任务
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids: Optional[bool] = None):
        """
        通过 Drop labels 的方式，实现 classifier-free guidance
        """
        if force_drop_ids is None:          # 随机丢弃标签
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:                               # 强制丢弃所有标签
            drop_ids = torch.BoolTensor(force_drop_ids is True)
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)    # 丢弃标签
        embeddings = self.embedding_table(labels)
        return embeddings


# 相当于 Transformer 中的 Feed Forward 层
class PointwiseFeedForward(nn.Module):
    """
    Pointwise Feed Forward，相当于普通的 MLP
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.net = Mlp(in_features, hidden_features, out_features, act_layer=act_layer, drop=drop)

    def forward(self, x):
        return self.net(x)


# 用于实现 AdaLN-Zero
def scale_and_shift(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# Diffusion Transformer 的 AdaLN-Zero Block
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning, namely DiT Block with adaLN-Zero
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        # 第一个块：LayerNorm -> Scale, Shift -> Attention -> Scale
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, **block_kwargs)

        # 第二个块：LayerNorm -> Scale, Shift -> Pointwise Feed Forward -> Scale
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.pff = PointwiseFeedForward(hidden_size, hidden_features=int(hidden_size * mlp_ratio),
                                        act_layer=approx_gelu, drop=0.)

        # AdaLN-Zero conditioning
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # Conditioning 经过 MLP 得到六个参数，分别是第一和第二个块的 shift, scale, gate(scale)
        shift_attn, scale_attn, gate_attn, shift_pff, scale_pff, gate_pff = self.adaLN_modulation(c).chunk(6, dim=-1)
        # LayerNorm -> Scale, Shift -> Attention -> Scale
        x = x + gate_attn.unsqueeze(1) * self.attn(scale_and_shift(self.norm1(x), shift_attn, scale_attn))
        # LayerNorm -> Scale, Shift -> Pointwise Feed Forward -> Scale
        x = x + gate_pff.unsqueeze(1) * self.pff(scale_and_shift(self.norm2(x), shift_pff, scale_pff))
        return x


# Diffusion Transformer 的最后一个 Block
class DiTFinalLayer(nn.Module):
    """
    DiT 的最后一层
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        # 最后一层：LayerNorm -> Scale, Shift -> Linear
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # Conditioning 经过 MLP 得到两个参数，分别是最后一层的 shift, scale
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        # LayerNorm -> Scale, Shift -> Linear
        x = scale_and_shift(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone
    """
    def __init__(
            self,
            input_size=32, patch_size=2,
            in_channels=4, hidden_size=1152, depth=28,
            num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, num_classes=1000,
            learn_sigma=True
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        # embeddings
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)    # patch 编码
        self.t_embedder = TimestepEmbedder(hidden_size)                                              # 噪声步进编码
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)                # 生成类别编码
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=True)  # 位置编码

        # transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = DiTFinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # 初始化 transformer blocks 中 nn.Linear 层的权重:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)   # 偏置初始化为 0
        self.apply(_basic_init)

        # 初始化（并冻结） pos_embed 为 sin-cos embeddings:
        pos_embed = get_2d_sinusoidal_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(pos_embed.float().unsqueeze(0))

        # 以 nn.Linear 的方式（而不是 nn.Conv2d）初始化 x_embedder:
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # 初始化 y_embedder:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # 初始化 t_embedder 的 mlp 层:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # 置零初始化 DiT blocks 中的 AdaLN 模块:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # 置零初始化最后一层的 AdaLN 模块和线性层:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor):
        """
        [N, H*W//(P*P), P*P*C] -> [N, H, W, C]

        Args:
            x: [N, H*W//(P*P), P*P*C]

        Returns:
            images: [N, H, W, C]
        """
        c = self.out_channels
        p: int = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]   # 确保 x 形状为 h, w 的整数倍数

        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('n h w p q c -> n c h p w q', x)
        images = x.reshape(x.shape[0], c, h * p, w * p)
        return images

    def forward(self, x, t, y):
        """
        Conditional Diffusion Transformer 前向传播

        Args:
            x: [N, C, H, W]
            t: [N]
            y: [N]

        Returns:
            logits: [N, out_channels, H, W]
        """
        # embeddings
        x = self.x_embedder(x) + self.pos_embed     # [N, T, D], where T = H*W/(P*P), D = P*P*C, where P = patch_size
        t = self.t_embedder(t)                      # [N, D]
        y = self.y_embedder(y, self.training)       # [N, D]

        # 条件融合
        c = t + y                                   # [N, D]

        # transformer blocks
        for block in self.blocks:
            x = block(x, c)                         # [N, T, D]
        x = self.final_layer(x, c)                  # [N, T, D_out], where D_out = P*P*out_channels
        x = self.unpatchify(x)                      # [N, out_channels, H, W]
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Unconditional Diffusion Transformer 前向传播
        参考代码：https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb

        Args:
            x: [N, C, H, W]
            t: [N]
            y: [N]
            cfg_scale: [N]

        Returns:
            logits: [N, C, H, W]
        """
        half_x = x[: len(x) // 2]
        repeat_x = torch.cat([half_x, half_x], dim=0)

        # 前向传播
        model_out = self.forward(repeat_x, t, y)

        # DiT 作者出于可复现性考虑，默认只对前三个通道进行 classifier-free guidance。
        # 标准的 cfg 方式会对所有通道进行 guidance。
        # 可以将第一行替换为下面这一行实现
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        conditional_eps, unconditional_eps = torch.split(eps, len(eps) // 2, dim=0)  # 也可以 torch.chunk(eps, 2, dim=0)
        half_eps = unconditional_eps + cfg_scale * (conditional_eps - unconditional_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


def get_2d_sinusoidal_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    Args:
        embed_dim: int of the embedding dimension
        grid_size: int of the grid size (grid_size x grid_size)
        cls_token: bool, if True, add a cls token to the embedding
        extra_tokens: int, number of extra tokens to add to the embedding (e.g. for padding)
    Returns:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')  # here w goes first
    grid = torch.stack(grid, dim=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sinusoidal_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = torch.cat([torch.zeros([extra_tokens, embed_dim]), pos_embed], dim=0)
    return pos_embed


def get_2d_sinusoidal_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    grid = grid.reshape(2, -1)
    emb_h = get_1d_sinusoidal_positional_embedding(grid[0], embed_dim // 2)  # [H*W, D//2], where D = embed_dim
    emb_w = get_1d_sinusoidal_positional_embedding(grid[1], embed_dim // 2)  # [H*W, D//2]

    emb = torch.cat(tensors=[emb_h, emb_w], dim=1)                           # [H*W, D]
    return emb


if __name__ == '__main__':
    model = DiT(input_size=32, patch_size=8, in_channels=4, hidden_size=384, depth=12, num_heads=6,
                mlp_ratio=4.0, class_dropout_prob=0.1, num_classes=1000, learn_sigma=True)
    x = torch.randn(2, 4, 32, 32)
    t = torch.randn(2)
    y = torch.randint(0, 1000, (2,))
    logits = model(x, t, y)
    print(logits.shape)
