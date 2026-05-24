import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from RoPE import apply_rotary_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    K, VのHead数をQのHead数に合わせるために複製(repeat)する関数
    入力x shape: (batch_size, seq_len, num_kv_heads, head_dim)
    出力 shape: (batch_size, seq_len, num_kv_heads * n_rep, head_dim)
    """

    batch_size, seq_len, num_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x

    # unsqueezeで次元を1つ増やし、expandで複製し、reshapeで元の次元数に戻す
    # 例：(B, S, KV_H, 1, D) -> (B, S, KV_H, n_rep, D) -> (B, S, KV_H * n_rep, D)
    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, num_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, num_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int = None):
        super().__init__()
        self.num_heads = num_heads

        # num_kv_heads が指定されていなければ、通常のMHA (num_kv_heads = num_heads) になる。
        # num_kv_headss = 1 なら MQA (Multi-Query Attention) になる
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

        # QのHead数とKVのHead数の比率 (グループサイズ)
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = dim // num_heads

        # Q は従来通り num_heads ぶん
        self.wq = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        # パラメータ数削減のため、K, V は num_kv_heads ぶんだけしか作らない。
        self.wk = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.num_heads * self.head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # 1. Q, K, V の計算
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # 2. ヘッドごとに分割
        # KとVは num_kv_heads で分割する
        xq = xq.view(batch_size, seq_len, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # 3. RoPEの適用
        xq, xk = self.apply_rope(xq, xk, freq_cis)

        # 4. GQAの適用
        # KとVのHead数が少ないため、QのHead数に合うように複製する。
        # xk, xv の shape は (batch_size, seq_len, num_heads, head_dim) になる。
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # 以降は通常の MHA と同じ処理
        # 5. 次元の入れ替え
        keys = xk.transpose(1, 2)
        queries = xq.transpose(1, 2)
        values = xv.transpose(1, 2)

        # 6. Attentionスコアの計算
        scores = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 7. マスクの適用
        if mask is not None:
            scores = scores + mask

        # 8. Softmaxの適用
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # 9. Value との合成
        output = torch.matmul(scores, values)

        # 10. 結合、出力
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        self.wo(output)
