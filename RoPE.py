import torch


# ベクトルを２次元ずつペアにして、それぞれのペアに異なる速度で回転を与える。
# 最初のペアは速く回転し、後半のペアになるほどゆっくり回転する (相対距離の減衰を表す)。
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    周波数 (frequencies) を事前に求める関数。
    - dim: 各ヘッドの次元数 (head_dim)
    - end: 最大シーケンス長 (最大位置)
    - theta: ベースの角度 (LLaMAでは基本 10000.0 ?)

    返り値：
    cos と sin を複素数 (complex 64)としてまとめたテンソル
    """

    # 1. 角度のベースとなる周波数を計算する
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # 2. トークン位置のテンソルを作成
    t = torch.arrange(end, device=freqs.device, dtype=torch.float32)  # type: ignore

    # 3. 位置 t と周波数 freqs の外積をとり、すべての位置・すべての次元ペアにおける角度を計算する
    # tのshape: (end, ), freqsのshape: (dim // 2, ) -> freqsのshape: (end, dim // 2) (// は切り捨て除算)
    freqs = torch.outer(t, freqs)  # torch. outerで外積をとれる

    # 4. 極座標系から複素数表現 (cos + i*sin) へ変換
    # e^(i * freqs) = cos(freqs) + i * sin(freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis


# 事前計算した複素数テンソル(freq_cis)を、入力x (QやK) のshapeに合わせてブロードキャストできるように変形する
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    事前計算した複素数テンソルを、入力x (QやK) のshapeに合わせてブロードキャストできるように変形する。
    x.shape: (batch_size, seq_len, num_heads, head_dim // 2) に対応させるため。
    """

    ndim = x.ndim
    assert 0 <= 1 < ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]

    """
    shape = []
    for i, d in enumerate(x.shape):
        if i == 1 or i == ndim - 1:
            shape.append(d)
        else:
            shape.append(1)
    """

    # shape = (1, seq_len, 1, head_dim // 2) になる
    return freqs_cis.view(*shape)


# QとKにRoPEを適用する
# QとKのshapeを合わせて、複素数テンソル(freqs_cis)をブロードキャスト可能な形に変形し、複素数の掛け算を行う
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    QとKにRoPEを適用する。
    xq, xk shape: (batch_size, seq_len, num_heads, head_dim)
    """

    # 1.最後の次元 (head_dim)を2つずつペアにするために、floatからcomplexに変換する
    # view_as_complexを使うと、最後の次元が半分になり、実部と虚部として扱われる
    # (batch_size, seq_len, num_heads, head_dim // 2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 2. freqs_cis をブロードキャスト可能なshapeに変換
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # 3. 複素数の掛け算をして、元の実数テンソルに戻す
    # (a + bi) * (c + di) = (ac - bd) * (ad + bc)i
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    # 元のデータ型 (fp16やbf16) に戻して返す
    return xq_out.type_as(xq), xk_out.type_as(xk)
