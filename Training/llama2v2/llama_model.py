# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.
from typing import Optional, Tuple, Type
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

device = 'cuda'

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 1024
    dim_k = None
    dim_v = None


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

"""
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            device=device
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            device=device
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            device=device
        )
        self.wo = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            device=device
        )

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim), device=device
        )
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim), device=device
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        print('xq size: ', xq.size()) # 1, 1000, 512
        print('bsz: ', bsz) # 1
        print('seqlen: ', seqlen)# 1000
        print('n local heads: ', self.n_local_heads) # 8
        print('self.head_dim: ', self.head_dim)# 64

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        #print('start pos: ', start_pos)
        #print('seqlen: ', seqlen)
        #print(self.cache_v.grad)
        print('cache k grad: ', self.cache_k.requires_grad)
        print('cache v grad: ', self.cache_v.requires_grad)
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        print('xq: ', xq.size()) # 1, 8, 1024, 64
        print('keys: ', keys.size()) #1, 8, 1024, 64
        print('values: ', values.size()) # 1, 8, 1024, 64
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        print('scores: ', scores.size()) # 1,8,1024,1024
        print('mask: ', mask.size()) #1,1,1024,1024
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)

"""  
class Attention(nn.Module):
    def __init__(
        self, args: ModelArgs
    ) -> None:
        super().__init__()

        assert args.dim % args.n_heads == 0, "n_heads must be a multiple of dim"

        if args.dim_k is None:
            args.dim_k = args.dim
        if args.dim_v is None:
            args.dim_v = args.dim

        self.max_seq_len = args.max_seq_len
        self.n_heads = args.n_heads
        self.dim_head = args.dim // args.n_heads
        self.dim_k = args.dim_k
        #self.causal = causal

        # positional encoding to be applied to query and key projections
        # self.positional_encoding = CosinePositionalEncoding(max_seq_len, dim // n_heads)
        #self.positional_encoding = RotaryPositionalEncoding(max_seq_len, dim // n_heads)

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
        #causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        #self.register_buffer("causal_mask", causal_mask)

    def forward(self, 
        x: torch.Tensor, 
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        return_scores: bool = False
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        # projects input to Q, K, V spaces
        q = self.proj_q(x)  # (bs, max_seq_len, dim_k)
        k = self.proj_k(x)  # (bs, max_seq_len, dim_k)
        v = self.proj_v(x)  # (bs, max_seq_len, dim_v)

        #print('q size: ', q.size())
        #print('bsz: ', bsz) # 1
        #print('seqlen: ', self.max_seq_len)# 1000
        #print('n local heads: ', self.n_heads) # 8
        #print('self.head_dim: ', self.dim_head)# 64
        # split projections between heads -> (bs, n_heads, max_seq_len, dim_k)
        q = q.view(bsz, self.max_seq_len, self.n_heads, self.dim_head)#.transpose(2, 3)
        k = k.view(bsz, self.max_seq_len, self.n_heads, self.dim_head)#.transpose(2, 3)
        v = v.view(bsz, self.max_seq_len, self.n_heads, self.dim_head)#.transpose(2, 3)

        # apply positional encoding to projections, for each heads
        #q = self.positional_encoding(q)  # (bs, n_heads, max_seq_len, dim_k)
        #k = self.positional_encoding(k)  # (bs, n_heads, max_seq_len, dim_k)
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        q = q.transpose(1, 2) # 1, 8, 1024, 64
        k = k.transpose(1, 2) # 1, 8, 1024, 64
        v = v.transpose(1, 2) # 1, 8, 1024, 64

        # Compute the correlation between a query q_i and all the keys, for every q_i
        #attn_scores = (q @ k.permute(0, 1, 3, 2)) * self.dim_k**-0.5  # (bs, n_heads, max_seq_len, max_seq_len)
        attn_scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.dim_head)

        # Fill the upper triangular part of the attention scores with -inf to inhibit them in the softmax
        #m_inf = -torch.finfo(attn_scores.dtype).max # TODO:what is this doing?
        #attn_scores.masked_fill_(mask[None, None, ...], m_inf)
        attn_scores = attn_scores + mask 

        # attention scores are used to build a weighted linear combination of values vectors
        attn_scores = torch.softmax(attn_scores, dim=-1)  # (bs, n_heads, max_seq_len, max_seq_len)
        out = attn_scores @ v  # (bs, n_heads, max_seq_len, dim_v)

        # merge heads
        out = out.transpose(2, 3).contiguous().view(bsz, self.max_seq_len, self.dim_k)  # (bs, max_seq_len, dim_v)

        # projects to the output space
        out = self.proj_out(out)  # (bs, max_seq_len, dim_v)

        return out
        #if return_scores:
        #    return out, attn_scores
        #else:
        #    return out
   

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
    def __init__(self, layer_id: int, args: ModelArgs):
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
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # print(params.vocab_size, params.dim)
        self.tok_embeddings = torch.nn.Embedding(params.vocab_size, params.dim, padding_idx=32000, device=device)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.output = nn.Linear(params.dim, params.vocab_size, bias=False, device=device)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    #TODO: make 2 versions?
    # @torch.inference_mode()
    def forward(self, tokens: torch.Tensor):
        start_pos = 0
        _bsz, seqlen = tokens.shape
        # print(f'tokens dim: {tokens.shape}\n')  # (1, 1000) == batch size x seq. length
        # print(tokens.max(), '\n\n', tokens)
        # print('\n\n')
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
        
        #hl = h#[:, -1, :]  # Probably for inference mode?

        #hl = hl.to(self.output.parameters().__next__().device)
        output = self.output(h)
        output = output.transpose(1, 2)
        
        return output.float()