from .Transformer import Transformer
import torch
from layers.transformer.TransformerDecoder import TransformerDecoder
from layers.transformer.TransformerEncoder import TransformerEncoder
from torch import nn


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 max_len: int,
                 d_model: int = 512,
                 heads_num: int = 8,
                 d_ff: int = 2048,
                 N: int = 6,
                 dropout_rate: float = 0.1,
                 layer_norm_eps: float = 1e-5,
                 pad_idx: int = 0,
                 device: torch.device = torch.device("cpu"),
                 ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.heads_num = heads_num
        self.d_ff = d_ff
        self.N = N
        self.dropout_rate = dropout_rate
        self.layer_norm_eps = layer_norm_eps
        self.pad_idx = pad_idx
        self.device = device

        self.encoder = TransformerEncoder(
            src_vocab_size,
            max_len,
            pad_idx,
            d_model,
            N,
            d_ff,
            heads_num,
            dropout_rate,
            layer_norm_eps,
            device,
        )

        self.decoder = TransformerDecoder(
            tgt_vocab_size,
            max_len,
            pad_idx,
            d_model,
            N,
            d_ff,
            heads_num,
            dropout_rate,
            layer_norm_eps,
            device,
        )

        self.linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """src,tgt:torch.Tensor 単語のid列"""

        # mask
        pad_mask_src = self._pad_mask(src)
        src = self.encoder(src, pad_mask_src)

        # target系列の"0(BOS)~max_len-1"(max_len-1系列)までを入力し、"1~max_len"(max_len-1系列)を予測する
        mask_self_attn = torch.logical_or(  # OR演算
            self._subsequent_mask(tgt), self._pad_mask(tgt)
        )
        dec_output = self.decoder(tgt, src, pad_mask_src, mask_self_attn)

        return self.linear(dec_output)

    def _pad_mask(self, x: torch.Tensor) -> torch.Tensor:
        """単語のid列からmaskを作成する"""
        seq_len = x.size(1)
        mask = x.eq(self.pad_idx)
        mask = mask.unsqueeze(1)
        mask = mask.repeat(1, seq_len, 1)
        return mask.to(self.device)

    def _subsequent_mask(self, x: torch.Tensor) -> torch.Tensor:
        """DecoderのMasked-Attentionに使用するmaskを作成する"""
        batch_size = x.size(0)
        max_len = x.size(1)
        return (
            torch.tril(torch.ones(batch_size, max_len, max_len)).eq(
                0).to(self.device)
        )
