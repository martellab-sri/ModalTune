# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
import os
import sys
import torch

this_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_file_dir, "../../"))

from torchscale.model import LongNetConfig as longnet_arch
from torchscale.architecture.config import EncoderConfig
from torchscale.architecture.decoder import Decoder, DecoderLayer
from torchscale.architecture.encoder import Encoder, EncoderLayer
from torchscale.component.dilated_attention import (
    DilatedAttention,
    DilatedAttentionLoraAdapter,
)
from fairscale.nn import checkpoint_wrapper, wrap


class LongNetDecoderLayer(DecoderLayer):

    def build_self_attention(self, embed_dim, args):
        return DilatedAttention(
            args,
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            encoder_decoder_attention=False,
            subln=args.subln,
        )


class LongNetDecoder(Decoder):

    def build_decoder_layer(
        self, args, depth, is_moe_layer=False, is_encoder_decoder=False
    ):
        layer = LongNetDecoderLayer(
            args,
            depth,
            is_moe_layer=is_moe_layer,
            is_encoder_decoder=is_encoder_decoder,
        )
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        if args.fsdp:
            layer = wrap(layer)
        return layer


class LongNetEncoderLayer(EncoderLayer):

    def build_self_attention(self, embed_dim, args):
        return DilatedAttention(
            args,
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            encoder_decoder_attention=False,
            subln=args.subln,
        )


class LongNetEncoder(Encoder):

    def build_encoder_layer(
        self, args, depth, is_moe_layer=False, is_encoder_decoder=False
    ):
        layer = LongNetEncoderLayer(
            args,
            depth,
            is_moe_layer=is_moe_layer,
            is_encoder_decoder=is_encoder_decoder,
        )
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        if args.fsdp:
            layer = wrap(layer)
        return layer


class LongNetLoraAdapterEncoderLayer(EncoderLayer):

    def build_self_attention(self, embed_dim, args):
        return DilatedAttentionLoraAdapter(
            args,
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            encoder_decoder_attention=False,
            subln=args.subln,
        )

    def forward(
        self,
        x,
        gene,
        task,
        encoder_padding_mask,
        attn_mask=None,
        rel_pos=None,
        multiway_split_position=None,
        incremental_state=None,
    ):
        if multiway_split_position is not None:
            assert self.args.multiway
            raise NotImplementedError
            # self.apply(set_split_position(multiway_split_position))

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
            gene = self.self_attn_layer_norm(gene)
            task = self.self_attn_layer_norm(task)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            gene=gene,
            task=task,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            rel_pos=rel_pos,
            incremental_state=incremental_state,
        )
        x = self.dropout_module(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        if not self.is_moe_layer:
            x = self.ffn(x)
            l_aux = None
        else:
            x = x.transpose(0, 1)
            x, l_aux = self.moe_layer(x)
            x = x.transpose(0, 1)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, l_aux


class LongNetLoraAdapterEncoder(Encoder):

    def build_encoder_layer(
        self, args, depth, is_moe_layer=False, is_encoder_decoder=False
    ):
        layer = LongNetLoraAdapterEncoderLayer(
            args,
            depth,
            is_moe_layer=is_moe_layer,
            is_encoder_decoder=is_encoder_decoder,
        )
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        if args.fsdp:
            layer = wrap(layer)
        return layer


def make_longnet(args):
    if args.arch in longnet_arch.__dict__.keys():
        longnet_args = longnet_arch.__dict__[args.arch]
    if hasattr(args, "dropout"):
        longnet_args["dropout"] = args.dropout
    if hasattr(args, "drop_path_rate"):
        longnet_args["drop_path_rate"] = args.drop_path_rate
    longnet_args = EncoderConfig(**longnet_args)
    model = LongNetEncoder(longnet_args)
    print(
        "Number of trainable LongNet parameters: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    return model


def make_longnet_from_name(
    config_name: str,
    dilated_ratio: str = "[1, 2, 4, 8, 16]",
    segment_length: str = "[1024, 2048, 4096, 8192, 16384]",
    drop_path_rate: int = 0.1,
    dropout: float = 0.1,
    lora_adapter: bool = False,
    lora_args: dict = None,
):
    """
    make LongNet model from config name

    Arguments:
    ----------
    config_name: str
        name of the config
    dilated_ratio: str
        dilated ratio
    segment_length: str
        segment length
    drop_path_rate: int
        drop path rate
    dropout: float
        dropout rate
    """
    if config_name in longnet_arch.__dict__.keys():
        longnet_args = longnet_arch.__dict__[config_name]

    longnet_args["dropout"] = dropout
    longnet_args["drop_path_rate"] = drop_path_rate

    # set dilated ratio and segment length
    longnet_args["dilated_ratio"] = dilated_ratio
    longnet_args["segment_length"] = segment_length

    print("dilated_ratio: ", dilated_ratio)
    print("segment_length: ", segment_length)

    if lora_adapter:
        assert (
            lora_args is not None
        ), "lora_args must be provided when lora_adapter is True"
        longnet_args.update(lora_args)

    longnet_args = EncoderConfig(**longnet_args)
    if lora_adapter:
        model = LongNetLoraAdapterEncoder(longnet_args)
    else:
        model = LongNetEncoder(longnet_args)
    print(
        "Number of LongNet parameters: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    return model
