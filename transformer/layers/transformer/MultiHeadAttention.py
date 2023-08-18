import torch
from layers.transformer.ScaledDotProductAttention import ScaledDotProductAttention
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.d_v = d_model // h

        self.W_k = nn.Parameter(
            torch.Tensor(h, d_model, self.d_k)  # ヘッド数、入力次元、出力次元(=入力次元/ヘッド数)
        )
        self.W_q = nn.Parameter(
            torch.Tensor(h, d_model, self.d_k)
        )
        self.W_v = nn.Parameter(
            torch.Tensor(h, d_model, self.d_v)
        )

        self.scaled_dot_product_attention = ScaledDotProductAttention(self.d_k)

        self.linear = nn.Linear(h*self.d_v, d_model)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask_3d: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, seq_len = q.size(0), q.size(1)

        # head数の分Query,Key,Valueを複製
        q = q.repeat(self.h, 1, 1, 1)  # head, batch_size,seq_len,d_model
        k = k.repeat(self.h, 1, 1, 1)  # head, batch_size, seq_len, d_model
        v = v.repeat(self.h, 1, 1, 1)  # head, batch_size, seq_len, d_model

        # Scaled Dot Product Attention前の線形変換
        q = torch.einsum(
            "hijk,hkl->hijl", (q, self.W_q)
        )  # head, batch_size, d_k, seq_len
        k = torch.einsum(
            "hijk,hkl->hijl", (k, self.W_k)
        )  # head, batch_size, d_k, seq_len
        v = torch.einsum(
            "hijk,hkl->hijl", (v, self.W_v)
        )  # head, batch_size, d_k, seq_len

        # ベクトルを並列に結合
        q = q.view(self.h*batch_size, seq_len, self.d_k)
        k = k.view(self.h*batch_size, seq_len, self.d_k)
        v = v.view(self.h*batch_size, seq_len, self.d_v)

        if mask_3d is not None:
            mask_3d = mask_3d.repeat(self.h, 1, 1)

        # Scaled Dot Product Attention
        # (self.h*batch_size, seq_len, d_model)
        attention_output = self.scaled_dot_product_attention(q, k, v, mask_3d)

        # １つのチャンクにつき(batch_size, seq_len, d_model),h個のチャンク
        attention_output = torch.chunk(attention_output, self.h, dim=0)

        # (batch_size, seq_len, d_model*self.h)
        attention_output = torch.cat(attention_output, dim=2)

        # 線形変換
        output = self.linear(attention_output)

        return output
