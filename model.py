import torch
import torch.nn as nn

from RMSNorm import RMSNorm
from RoPE import precompute_freqs_cis
from TransformerBlock import TransformerBlock


class Llama(nn.Module):
    def __init__(
        self,
        vocab_size: int,  # 語彙サイズ
        dim: int,  # 隠れ層の次元数
        hidden_dim: int,  # FeedForward層の次元数
        num_heads: int,  # Qのヘッド数
        num_kv_heads: int,  # KVのヘッド数 (GQA用)
        n_layers: int,  # TransformerBlockの層数
        max_seq_len: int,  # 最大トークン長
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # 1. Token Embedding 層
        # 単語IDを dim 次元のベクトルに埋め込む
        self.token_embedding = nn.Embedding(vocab_size, dim)

        # 2. TransformerBlock を n_layers 層分繰り返す
        self.leyers = nn.ModuleList(
            [
                TransformerBlock(dim, hidden_dim, num_heads, num_kv_heads, norm_eps)
                for _ in range(n_layers)
            ]
        )

        # 3. 最後の正規化層
        self.norm = RMSNorm(dim, norm_eps)

        # 4. 出力層 (LM Head)
        # 最終的なベクトルから、vocab_size 個の単語のどれが次に来るのかのスコアを計算。
        # LLaMAでは bias=False
        self.output = nn.Linear(dim, vocab_size, bias=False)

        # 5. RoPE の事前計算 (学習パラメータではないため、register_buffer で保存しておく)
        # dim // num_heads は head_dim (Head 1つあたりの次元数)
        freqs_cis = precompute_freqs_cis(dim // num_heads, max_seq_len * 2)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        tokens shape: (batch_size, seq_len) 中身は単語ID
        """
        batch_size, seq_len = tokens.shape
        assert seq_len <= self.max_seq_len, (
            f"入力シーケンス長({seq_len})が最大長({self.max_seq_len})を超えています"
        )

        # 1. 単語IDをベクトルに変換
        # shape: (batch_size, seq_len, dim)
        h = self.token_embedding(tokens)

        # 2. RoPEの取得 (必要な長さだけスライドして取得)
        freqs_cis = self.freqs_cis[:seq_len]

        # 3. Causal Maskの生成
        # 望ましい形は右上半分が -∞、左下半分が 0の行列
        mask = None
        if seq_len > 1:
            # triu は上三角行列を作る。diagonal=1 で、対角線の1つ上からを対象にする
            mask = torch.full((seq_len, seq_len), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            # maskのshapeをブロードキャスト用に合わせる: (1, 1, seq_len, seq_len)
            mask = mask.unsqueeze(0).unsqueeze(0)

        # 4. Transformer Block を順番に通過させる
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)

        # 5. 最後の正規化
        h = self.norm(h)
        # 6. 出力層を通して、語彙のlogitsを出力
        # shape: (batch_size, seq_len, vocab_size)
        logits = self.output(h)

        return logits
