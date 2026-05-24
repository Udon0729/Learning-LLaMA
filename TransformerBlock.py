# LLaMAは処理の前に正規化(RMSNorm)を行うPre-Layer Normalization
# Attention is all you needではPost-Layer Normalization

import torch
import torch.nn as nn

from Attention import Attention
from FeedForward import FeedForward
from RMSNorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.attention = Attention(dim, num_heads, num_kv_heads)
        self.feed_forward = FeedForward(dim, hidden_dim)

        # 各モジュールの前で適用する正規化層
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        # 1. Attentionブロック (残差接続)
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)

        # 2. FeedForwardブロック (残差接続)
        out = h + self.feed_forward(self.ffn_norm(h))

        return out
