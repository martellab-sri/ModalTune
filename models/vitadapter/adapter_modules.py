# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
"""
Code adapted from https://github.com/czczup/ViT-Adapter to better suit ModalTune
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from .drop_path import DropPath


class SelfAttentionLayer(nn.Module):
    """
    Simple SelfAttentionLayer
    from mask2former: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py#L333
    """

    def __init__(
        self,
        d_model,
        nheads,
        dropout=0.0,
        normalize_before=False,
        with_cffn=False,
        cffn_ratio=1.0,
    ):
        super().__init__()
        self.with_cffn = with_cffn
        self.cffn_ratio = cffn_ratio
        embed_model = d_model
        if self.with_cffn:
            embed_model = int(d_model * cffn_ratio)
            self.q_proj = nn.Linear(d_model, embed_model)
            self.output_proj = nn.Linear(embed_model, d_model)

        self.self_attn = nn.MultiheadAttention(
            embed_model,
            nheads,
            dropout=dropout,
            batch_first=True,
            kdim=d_model,
            vdim=d_model,
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.functional.relu
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, query_pos: Optional[torch.Tensor] = None):
        query = k = self.with_pos_embed(tgt, query_pos)
        if self.with_cffn:
            query = self.q_proj(query)

        tgt2 = self.self_attn(query, k, value=tgt)[0]

        if self.with_cffn:
            tgt2 = self.output_proj(tgt2)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, query_pos: Optional[torch.Tensor] = None):
        tgt2 = self.norm(tgt)
        query = k = self.with_pos_embed(tgt2, query_pos)
        if self.with_cffn:
            query = self.q_proj(query)

        tgt2 = self.self_attn(query, k, value=tgt2)[0]

        if self.with_cffn:
            tgt = tgt + self.dropout(self.output_proj(tgt2))
        else:
            tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, query_pos: Optional[torch.Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, query_pos)
        return self.forward_post(tgt, query_pos)


class Identity_mod(nn.Module):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, input, *args, **kwargs):
        return input


class CrossAttentionLayer(nn.Module):
    """
    Simple CrossAttentionLayer
    from mask2former: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py#L333

    Change log:
    Removed residual connection from cross attention layer
    """

    def __init__(
        self,
        d_model,
        nheads,
        dropout=0.0,
        normalize_before=False,
        with_cffn=False,
        cffn_ratio=1.0,
    ):
        super().__init__()
        self.with_cffn = with_cffn
        self.cffn_ratio = cffn_ratio
        embed_model = d_model
        if self.with_cffn:
            embed_model = int(d_model * cffn_ratio)
            self.q_proj = nn.Linear(d_model, embed_model)
            self.output_proj = nn.Linear(embed_model, d_model)

        self.multihead_attn = nn.MultiheadAttention(
            embed_model,
            nheads,
            dropout=dropout,
            batch_first=True,
            kdim=d_model,
            vdim=d_model,
        )

        if normalize_before:
            self.norm_kq = nn.LayerNorm(d_model)
        self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.activation = nn.functional.relu
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ):

        if self.with_cffn:
            query = self.q_proj(self.with_pos_embed(tgt, query_pos))
        else:
            query = self.with_pos_embed(tgt, query_pos)

        tgt2 = self.multihead_attn(
            query=query,
            key=self.with_pos_embed(memory, pos),
            value=self.with_pos_embed(memory, pos),
        )[0]
        if self.with_cffn:
            tgt2 = self.output_proj(tgt2)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ):
        tgt2 = self.norm(tgt)
        memory = self.norm_kq(memory)

        if self.with_cffn:
            query = self.q_proj(self.with_pos_embed(tgt2, query_pos))
        else:
            query = self.with_pos_embed(tgt2, query_pos)

        tgt2 = self.multihead_attn(
            query=query,
            key=self.with_pos_embed(memory, pos),
            value=self.with_pos_embed(memory, pos),
        )[0]
        if self.with_cffn:
            tgt = tgt + self.dropout(self.output_proj(tgt2))
        else:
            tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, pos, query_pos)
        return self.forward_post(tgt, memory, pos, query_pos)


class FFNLayer(nn.Module):
    """
    Simple feed forward network
    from mask2former: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py#L333
    """

    def __init__(
        self, d_model, dim_feedforward=256, dropout=0.0, normalize_before=False
    ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = nn.functional.relu
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        return tgt2

    def forward(self, tgt, pos=None):
        tgt = self.with_pos_embed(tgt, pos)
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class Extractor(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        with_cffn=True,
        cffn_ratio=0.25,
        drop=0.0,
        drop_path=0.0,
        with_cp=False,
    ):
        super().__init__()
        self.attn = CrossAttentionLayer(
            d_model=dim,
            nheads=num_heads,
            normalize_before=True,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
        )
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = FFNLayer(dim, int(dim * cffn_ratio), drop, True)
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, query, feat, pos=None):
        def _inner_forward(query, feat, pos):
            attn = self.attn(query, feat, None, pos)
            query = query + attn

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(query))
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat, pos)
        else:
            query = _inner_forward(query, feat, pos)

        return query


class Injector(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        init_values=0.0,
        with_cp=False,
        with_cffn=True,
        cffn_ratio=0.25,
    ):
        super().__init__()
        self.with_cp = with_cp
        self.attn = CrossAttentionLayer(
            d_model=dim,
            nheads=num_heads,
            normalize_before=True,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
        )
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, feat, pos=None):
        def _inner_forward(query, feat, pos):
            attn = self.attn(query, feat, pos, None)
            return query + self.gamma * attn

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat, pos)
        else:
            query = _inner_forward(query, feat, pos)

        return query


class InteractionBlockWithCls(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        drop=0.0,
        drop_path=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        init_values=0.0,
        extra_extractor=False,
        with_cp=False,
    ):
        super().__init__()

        self.injector = Injector(
            dim=dim,
            num_heads=num_heads,
            init_values=init_values,
            with_cp=with_cp,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
        )
        self.extractor = Extractor(
            dim=dim,
            num_heads=num_heads,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            drop=drop,
            drop_path=drop_path,
            with_cp=with_cp,
        )
        if extra_extractor:
            self.extra_extractors = nn.Sequential(
                *[
                    Extractor(
                        dim=dim,
                        num_heads=num_heads,
                        with_cffn=with_cffn,
                        cffn_ratio=cffn_ratio,
                        drop=drop,
                        drop_path=drop_path,
                        with_cp=with_cp,
                    )
                    for _ in range(2)
                ]
            )
        else:
            self.extra_extractors = None

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, x, c, cls, blocks, H_toks, W_toks, query_pos=None):
        x = self.injector(
            query=x,
            feat=c,
            pos=query_pos,
        )
        x = torch.cat((cls, x), dim=1)
        for idx, blk in enumerate(blocks):
            x = blk(x, H_toks, W_toks)
        cls, x = (
            x[
                :,
                :1,
            ],
            x[
                :,
                1:,
            ],
        )
        c = self.extractor(
            query=c,
            feat=x,
            pos=query_pos,
        )
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(
                    query=c,
                    feat=x,
                    pos=query_pos,
                )
        return x, c, cls


class InteractionBlockWithCls_LongNetViT(InteractionBlockWithCls):
    def __init__(
        self,
        dim,
        num_heads=6,
        drop=0,
        drop_path=0,
        with_cffn=True,
        cffn_ratio=0.25,
        init_values=0,
        extra_extractor=False,
        with_cp=False,
    ):
        super().__init__(
            dim,
            num_heads,
            drop,
            drop_path,
            with_cffn,
            cffn_ratio,
            init_values,
            extra_extractor,
            with_cp,
        )

    def forward(
        self, x, c, cls, blocks, incremental_state, layer_configs, query_pos=None
    ):
        x = self.injector(
            query=x,
            feat=c,
            pos=query_pos,
        )
        x = torch.cat((cls, x), dim=1)
        for idx, blk in enumerate(blocks):
            x, _ = blk(
                x,
                incremental_state=(
                    incremental_state[idx] if incremental_state is not None else None
                ),
                **layer_configs
            )
        cls, x = (
            x[
                :,
                :1,
            ],
            x[
                :,
                1:,
            ],
        )
        c = self.extractor(
            query=c,
            feat=x,
            pos=query_pos,
        )
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(
                    query=c,
                    feat=x,
                    pos=query_pos,
                )
        return x, c, cls


class InteractionBlockWithCls_TITAN(InteractionBlockWithCls):
    def forward(self, x, c, cls, blocks, attn_bias, bg_mask, query_pos=None):
        x = self.injector(
            query=x,
            feat=c,
            pos=query_pos,
        )
        x = torch.cat((cls, x), dim=1)
        for idx, blk in enumerate(blocks):
            x = blk(x, attn_bias, bg_mask)
        cls, x = (
            x[
                :,
                :1,
            ],
            x[
                :,
                1:,
            ],
        )
        c = self.extractor(
            query=c,
            feat=x,
            pos=query_pos,
        )
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(
                    query=c,
                    feat=x,
                    pos=query_pos,
                )
        return x, c, cls
