import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # ゼロ除算を防ぐための極小の値
        self.eps = eps
        # 学習対象のパラメータを定義
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 計算をfp32で行うためキャストする
        x_fp32 = x.float()

        # 1. 各要素を二乗する → x.pows(2)
        # 2. 最後の次元 (-1) で平均をとる → .mean(-1, keepdim=True) keepdim=Trueを忘れると、次元が潰れてしまって、除算時にエラーになる。
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        normed = x_fp32 * torch.rsqrt(variance + self.eps)

        # 3. 平方根をとって eps を加算し、逆数を掛ける。→torch.sqrt()の逆数は、torch.rsqrt()がよい
        # 4. 学習パラメータを掛けてスケールする。
        # 元のデータ型(fp16やbf16だったかもしれない)に戻してから、weightを掛ける
        return normed.type_as(x) * self.weight
