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
