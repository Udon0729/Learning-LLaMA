import torch
import torch.nn as nn
import torch.nn.functional as F

# SwiGLU (Swish-Gated Linear Unit) は、LLaMAで使用されるフィードフォワードネットワークの活性化関数。
# 重み行列が3つあるため、Transformerのデフォルトである隠れ次元の決め方である4 * dimは使用しない。8/3 * dim程度にして、さらに256の倍数に切り上げている。


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()

        # LLaMAでは、すべての線形層でbiasを使用しない。計算がシンプルとなり、学習が安定する。
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Gate projectioin
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # Down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # Up projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. xをw1に通して SiLUを適用
        # 2. xをw3に通す
        # 3. 1と2を掛け合わせて、最後に2に通す

        # SiLU関数 (Sigmoid Linear Unit)は精度面で主流だが、スパース性を生まないため、CPU推論時の高速化が難しい。
        # ReLU関数は負値をすべて0にするため、高いスパース性を得られるが、負値の入力で勾配が0になるため、dead neuron 問題が発生し学習がすすまない。
        # → Stochastic activations という手法があるらしい。

        # F.silu()はPyTorchによるものだが、数学的にはLLaMA論文で言及されるswitch関数と同じもの。
        return self.w2(F.silu(self.w1(x) * self.w3(x)))
