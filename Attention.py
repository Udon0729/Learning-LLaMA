# LLaMA本来のGQAやMQAではなく、単純なSelf-Attention

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# RoPE.pyからapply_rotary_embをインポート
from RoPE import apply_rotary_emb


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // num_heads

        # LLaMAではbiasを使用しない
        self.wq = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_heads * self.head_dim, dim, bias=False)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        # 入力xのshape: (batch_size, seq_len, dim)
        batch_size, seq_len, _ = x.shape

        # 1. Q, K, Vの計算
        # 各shape: (batch_size, seq_len, dim)
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # 2. Headごとに分割するためにreshapeする
        # shape: (batch_size, seq_len, num_heads, head_dim)
        xq = xq.view(batch_size, seq_len, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.num_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 3. Q と K に RoPE を適用
        # 出力shapeは変わらない (batch_size, seq_len, num_heads, head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # 4. Attention スコアの計算のために次元を入れ替える
        # shapeを (batch_size, num_heads, seq_len, head_dim) に変更
        # transpose(1, 2) で seq_len と num_heads の位置を入れ替える
        keys = xk.transpose(1, 2)
        queries = xq.transpose(1, 2)
        values = xv.transpose(1, 2)

        # 5. Q と K^T の内積 (Scaled Dot-Product)
        # queries: (batch_size, num_heads, seq_len, head_dim)
        # keys.transpose(2, 3): (batch_size, num_heads, head_dim, seq_len)
        # scores shape: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 6. マスクの適用 (Causal Maskingなど)
        if mask is not None:
            # maskが適用される部分は、-∞にして、Softmaxで0になるようにする
            scores = scores + mask

        # 7. Softmaxをかけて確率 (Attention weights) に変換
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # 8. Attention weights と V を掛け合せる
        # scores: (batch_size, num_heads, seq_len, seq_len)
        # values: (batch_size, num_heads, seq_len, head_dim)
        # output shape: (batch_size, num_heads, seq_len, head_dim)
        output = torch.matmul(scores, values)

        # 9. すべてのヘッドを統合して、元の次元に戻す
        # transposeで (batch_size, seq_len, num_heads, head_dim)に戻す
        # contiguous().view()で、(batch_size, seq_len, dim)に平坦化する
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # 10. 最後の出力層 (wo) に通す
        # shape: (batch_size, seq_len, dim)
        return self.wo(output)
