import numpy as np
import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int) -> None:
        super().__init__()
        self.d_k = d_k

    def forward(self,
                q: torch.Tensor,  # クエリ(N*D)(N個の単語を持つ文章をD次元の分散表現で表現)
                k: torch.Tensor,  # キー(N*D)
                v: torch.Tensor,  # バリュー
                mask: torch.Tensor = None,
                ) -> torch.Tensor:
        scalar = np.sqrt(self.d_k)  # √D
        attention_weight = torch.matmul(  # matmul:内積
            q, torch.transpose(k, 1, 2))/scalar  # Q*X^T/(D^0.5)の計算 transpose:転置 2個目の軸と3個目の軸を入れ替え
        if mask is not None:  # maskに対する処理
            if mask.dim() != attention_weight.dim():
                raise ValueError(
                    "mask.dim != attention_weight.dim, mask.dim={}, attention_weight.dim={}".format(
                        mask.dim(), attention_weight.dim()
                    )
                )
            attention_weight = attention_weight.data.masked_fill_(
                mask, -torch.finfo(torch.float).max
            )
        attention_weight = nn.functional.softmax(attention_weight, dim=2)
        return torch.matmul(attention_weight, v)  # (Attention Weight)*Xにより重みづけ
