# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.
from typing import Optional, Tuple, Type
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from config import train_config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

#TODO: clean all these classes up, add comments
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(
        self, args: train_config
    ) -> None:
        super().__init__()

        assert args.dim % args.n_heads == 0, "n_heads must be a multiple of dim"

        if args.dim_k is None:
            args.dim_k = args.dim
        if args.dim_v is None:
            args.dim_v = args.dim

        self.seq_len = args.seq_len
        self.n_heads = args.n_heads
        self.dim_head = args.dim // args.n_heads
        self.dim_k = args.dim_k
        #self.causal = causal

        # positional encoding to be applied to query and key projections
        # self.positional_encoding = CosinePositionalEncoding(seq_len, dim // n_heads)
        # self.positional_encoding = RotaryPositionalEncoding(seq_len, dim // n_heads)

        # Query, Key and Value projections
        self.proj_q = nn.Linear(args.dim, args.n_heads * self.dim_head, bias=False, device=device)
        self.proj_k = nn.Linear(
            args.dim,
            args.n_heads * self.dim_head,
            bias=False,
            device=device
        )
        self.proj_v = nn.Linear(
            args.dim,
            args.n_heads * self.dim_head,
            bias=False,
            device=device
        )
        self.proj_out = nn.Linear(
            args.dim,
            args.n_heads * self.dim_head,
            bias=False,
            device=device
        )

        # Build the causal mask, masking upper triangular part of attention scores
        #causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        #self.register_buffer("causal_mask", causal_mask)

    def forward(self, 
        x: torch.Tensor, 
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        #print('x shape: ', x.size())

        # projects input to Q, K, V spaces
        q = self.proj_q(x)  # (bs, seq_len, dim_k)
        k = self.proj_k(x)  # (bs, seq_len, dim_k)
        v = self.proj_v(x)  # (bs, seq_len, dim_v)

        #print('q size: ', q.size()) # 1, 1023, 512 NEEDS to be 1, 1023, 512
        #print('bsz: ', bsz) # 1
        #print('seqlen: ', seqlen) # 1024
        #print('n local heads: ', self.n_heads) # 8
        #print('self.head_dim: ', self.dim_head) # 64
        # split projections between heads -> (bs, n_heads, seq_len, dim_k)
        q = q.view(bsz, seqlen, self.n_heads, self.dim_head)#.transpose(2, 3)
        k = k.view(bsz, seqlen, self.n_heads, self.dim_head)#.transpose(2, 3)
        v = v.view(bsz, seqlen, self.n_heads, self.dim_head)#.transpose(2, 3)

        # apply positional encoding to projections, for each heads
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        q = q.transpose(1, 2) # 1, 8, 1024, 64
        k = k.transpose(1, 2) # 1, 8, 1024, 64
        v = v.transpose(1, 2) # 1, 8, 1024, 64

        # Compute the correlation between a query q_i and all the keys, for every q_i
        #attn_scores = (q @ k.permute(0, 1, 3, 2)) * self.dim_k**-0.5  # (bs, n_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.dim_head)

        attn_scores = attn_scores + mask 

        # attention scores are used to build a weighted linear combination of values vectors
        attn_scores = torch.softmax(attn_scores, dim=-1)  # (bs, n_heads, seq_len, seq_len)
        out = attn_scores @ v  # (bs, n_heads, seq_len, dim_v)

        out = out.transpose(2, 3).contiguous().view(bsz, seqlen, self.dim_k)  # (bs, seq_len, dim_v)

        # projects to the output space
        out = self.proj_out(out)  # (bs, seq_len, dim_v)

        return out
   

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
            device=device
        )
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
            device=device
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: train_config):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: train_config):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = torch.nn.Embedding(params.vocab_size, params.dim, padding_idx=params.pad_tok, device=device)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.output = nn.Linear(params.dim, params.vocab_size, bias=False, device=device)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.seq_len * 2
        )

    def forward(self, tokens: torch.Tensor):
        start_pos = 0
        _bsz, seqlen = tokens.shape
        # print(f'tokens dim: {tokens.shape}\n')  # (1, 1000) == batch size x seq. length
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        
        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = h.to(layer.parameters().__next__().device)
            h = layer(h, start_pos, freqs_cis, mask)
        h = h.to(self.norm.parameters().__next__().device)
        h = self.norm(h)

        output = self.output(h)
        output = output.transpose(1, 2)
        
        return output.float()