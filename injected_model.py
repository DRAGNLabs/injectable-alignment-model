# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.
from typing import Optional, Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F
from irm import NPI

class RMSNorm(torch.nn.Module):
    """
    Normalization module. RMSNorm (Root Mean Square Layer Normalization) is a form of normalization that is more computationally efficient than LayerNorm. 
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))


    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions. Used by rotary embeddings, instead of typical positional embeddings.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
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
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    """
    Multi-head attention module.
    """
    def __init__(
        self, args
    ) -> None:
        super().__init__()

        assert args.dim % args.n_heads == 0, "n_heads must be a multiple of dim"

        if args.dim_k is None:
            args.dim_k = args.dim
        if args.dim_v is None:
            args.dim_v = args.dim

        self.sequence_length = args.sequence_length
        self.n_heads = args.n_heads
        self.dim_head = args.dim // args.n_heads
        self.dim_k = args.dim_k
        #self.causal = causal

        # Query, Key and Value projections
        self.proj_q = nn.Linear(args.dim, args.n_heads * self.dim_head, bias=False)
        self.proj_k = nn.Linear(
            args.dim,
            args.n_heads * self.dim_head,
            bias=False
        )
        self.proj_v = nn.Linear(
            args.dim,
            args.n_heads * self.dim_head,
            bias=False
        )
        self.proj_out = nn.Linear(
            args.dim,
            args.n_heads * self.dim_head,
            bias=False
        )

        # Build the causal mask, masking upper triangular part of attention scores
        #causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        #self.register_buffer("causal_mask", causal_mask)

    def forward(self, 
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        # projects input to Q, K, V spaces
        q = self.proj_q(x)  # (bs, seq_len, dim_k)
        k = self.proj_k(x)  # (bs, seq_len, dim_k)
        v = self.proj_v(x)  # (bs, seq_len, dim_v)

        # split projections between heads -> (bs, n_heads, seq_len, dim_k)
        q = q.view(bsz, seqlen, self.n_heads, self.dim_head)
        k = k.view(bsz, seqlen, self.n_heads, self.dim_head)
        v = v.view(bsz, seqlen, self.n_heads, self.dim_head)

        # apply positional encoding to projections, for each heads
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        q = q.transpose(1, 2) 
        k = k.transpose(1, 2) 
        v = v.transpose(1, 2) 

        # Compute the correlation between a query q_i and all the keys, for every q_i
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

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False
        )
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args):
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
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention.forward(
            self.attention_norm(x), freqs_cis, mask
        )
      
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        if self.layer_id == 8:
            irm_output = NPI.forward(x)
            out += irm_output
        return out


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.IRM = NPI()

        # NOTE: Original Llama2 was not using padding, so did not use padding_idx. Will not work if tokenizer is trained without padding (disabled by default)
        self.embedding_encoder = torch.nn.Embedding(config.vocab_size, config.dim,  padding_idx=config.pad_id)  #

        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(TransformerBlock(layer_id, config))

        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096. 
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            config.dim // config.n_heads, config.sequence_length * 2
        )

    def forward(self, tokens: torch.Tensor):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        _bsz, seqlen = tokens.shape
        # Embed tokens
        h = self.embedding_encoder(tokens)

        # Grabs necessary precomputed frequencies, used for rotary embeddings during attention, which is used instead of typical positional embedding
        self.freqs_cis = self.freqs_cis.to(tokens.device)
        freqs_cis = self.freqs_cis[:seqlen]
        
        # Generate attention mask
        mask = None
        if seqlen > 1:
            # Fill mask with -inf
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf")
            )
            
            # Unmask across diagonal 1, meaning center diagonal is unmasked
            mask = torch.triu(mask, diagonal=1).type_as(h)

        # Iterate through all layers
        for layer in self.layers:
            #h = h.to(layer.parameters().__next__().device) # TODO: don't think we need these lines? may be important for parallelization
            h = layer(h, freqs_cis, mask)
        
        #h = h.to(self.norm.parameters().__next__().device)
        h = self.norm(h)

        output = self.output(h)

        output = output.transpose(1, 2) # transposes to batch_size, vocab_size, sequence_length
        
        return output.float()