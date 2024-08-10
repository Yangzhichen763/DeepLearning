
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from modules.activation import Swish
from embedding import Embedding


class TimeEmbeddingProjection(Embedding):
    """
    Time embedding projection module.
    """
    # noinspection PyPep8Naming
    def __init__(self, T, in_dim, embedding_dim, hidden_dim=None,
                 activation=Swish(), **embedding_kwargs):
        assert embedding_dim % 2 == 0

        super().__init__()
        position = torch.arange(T).float()                  # [T]
        self.embedding = get_timestep_embedding(position, embedding_dim)

        hidden_dim = hidden_dim or embedding_dim
        self.time_embedding = nn.Sequential(
            nn.Embedding.from_pretrained(self.embedding, **embedding_kwargs),
            nn.Linear(in_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        embedding = self.time_embedding(t)
        return embedding


def get_timestep_embedding(timesteps, embedding_dim):
    """
    Args:
        timesteps: [T]
        embedding_dim (int): the dimension of the embedding vector
    """
    assert len(timesteps.shape) == 1

    position = timesteps.float()                        # [T]
    embedding = torch.exp(
        torch.arange(0, embedding_dim, step=2, device=position.device, dtype=torch.float32)
        / embedding_dim * -math.log(10000)
    )                                                   # [d_model // 2]
    embedding = position[:, None] * embedding[None, :]  # [T, 1] * [1, d_model // 2] -> [T, d_model // 2]
    embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=-1)

    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1, 0, 0))

    return embedding

