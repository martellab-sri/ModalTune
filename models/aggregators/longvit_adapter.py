"""
Complete pipeline for ModalTune with Prov-Gigapath slide encoder integrated with ModalAdapters
"""

import math
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vitadapter.adapter_modules import (
    InteractionBlockWithCls_LongNetViT,
    SelfAttentionLayer,
    CrossAttentionLayer,
    Identity_mod,
)
from ..prov_gigapath.gigapath.slide_encoder import LongNetViT
from ..genomic_utils import GeneBaseClass
from .aggregators import Aggregator

from model_configs.other_configs import set_genomic_config
from utils.constants import GIGAPATH_WEIGHT_LOC


@Aggregator.register("longnetvit_gene_adapter")
class LongNetGeneAdapter(LongNetViT, Aggregator):
    """
    LongNet ViT Adapter with Gene Adapters and task prompt
    """
    def __init__(
        self,
        num_heads=12,
        gene_group_defination={},
        geneclass_name="gene_mixer_group",
        output_dim=256,
        init_values=0.0,
        interaction_indexes=None,
        with_cffn=True,
        cffn_ratio=0.25,
        add_prompt_feature=True,
        use_extra_extractor=True,
        freeze_vit=True,
        with_cp=False,
        use_prompt_sa=False,
        prompt_dropout=0.0,
        prompt_agg="cls",
        token_agg="cat",
        pretrained=True,
        multi_task=1,
        **kwargs
    ):
        """
        Parameters:
            interaction_indexes: List[List[int,int]], layers at which injectors and extractors would interact
            with_cffn: Set true to enable compression of feature space in attention layers
            cffn_ratio: Ratio of compression
            add_prompt_feature: Set true to add prompt feature to imaging feature
            use_extra_extractor: Set true to use extra extractors at the end of extractor step
            with_cp: Set true for checkpointing, see checkpoint for more details
            use_prompt_sa: Set true for using self attention
            prompt_dropout: dropout rate to be used in self attention of additional prompts
            prompt_agg: Choose out of cls/avg for getting final feature vector from gene tokens step
            token_agg: Choose out of cat/sum for how to combine final feature vectors from different modalities
            pretrained: Set true for loading the aggregator weights
            weights_location: Location for weights of the aggregator
            multitask: int, number of different embeddings for different tasks to calculate given same input
        """
        LongNetViT.__init__(self, **kwargs)
        # Load parameters after init
        self.load_slide_encoder(
            pretrained=pretrained, weights_location=GIGAPATH_WEIGHT_LOC
        )
        if freeze_vit:
            for name, param in self.named_parameters():
                param.requires_grad = False

        self.mode = "feature"
        self.num_block = self.depth
        self.interaction_indexes = interaction_indexes
        self.add_prompt_feature = add_prompt_feature
        self.prompt_agg = prompt_agg
        self.token_agg = token_agg
        self.is_multi = multi_task > 1
        embed_dim = self.embed_dim

        block_fn = InteractionBlockWithCls_LongNetViT

        self.interactions = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    init_values=init_values,
                    drop_path=self.drop_path_rate,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    extra_extractor=(
                        (True if i == len(interaction_indexes) - 1 else False)
                        and use_extra_extractor
                    ),
                    with_cp=with_cp,
                )
                for i in range(len(interaction_indexes))
            ]
        )

        self.prompt_selfattention = nn.Sequential(
            Identity_mod(),
            *[
                (
                    SelfAttentionLayer(
                        d_model=embed_dim,
                        nheads=num_heads,
                        dropout=prompt_dropout,
                        normalize_before=True,
                        with_cffn=with_cffn,
                        cffn_ratio=cffn_ratio,
                    )
                    if use_prompt_sa
                    else Identity_mod()
                )
                for _ in range(1, len(interaction_indexes))
            ]
        )

        num_gene_groups = len(gene_group_defination)
        args = argparse.Namespace(geneclass_name=geneclass_name, num_classes=2)
        print("Genomic model configurations: {}".format(set_genomic_config(args)))
        self.gene_encoder = GeneBaseClass.create(
            subclass_name=geneclass_name,
            **set_genomic_config(args),
            output_dim=embed_dim,
            mode="feature",
            group_sizes=gene_group_defination,
            n_groups=num_gene_groups
        )

        # to account for any changes based on cls token
        num_gene_groups = self.gene_encoder.n_groups
        # adding cls token
        if self.prompt_agg == "cls":
            self.gene_cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
            num_gene_groups = num_gene_groups + 1
            nn.init.trunc_normal_(self.gene_cls.data, std=0.02)

        # gene positional embeddings, add 1 for prompt token
        self.gene_pe = nn.Parameter(
            torch.zeros(num_gene_groups + int(self.is_multi), embed_dim)
        )

        # multiple tasks
        # Converts NUM_TASKS tasks into promptable input
        if self.is_multi:
            self.task_weight = nn.Sequential(
                *[nn.Linear(multi_task, embed_dim), nn.LayerNorm(embed_dim)]
            )
            self.task_weight.apply(self._init_weights)

        if self.token_agg == "cat":
            self.final_norm = nn.LayerNorm((2 + int(self.is_multi)) * embed_dim)
            self.final_project = nn.Linear(
                (2 + int(self.is_multi)) * embed_dim, output_dim
            )
        elif self.token_agg == "sum":
            self.final_norm = nn.LayerNorm(embed_dim)
            self.final_project = nn.Linear(embed_dim, output_dim)
        else:
            raise NotImplementedError

        # Initializations
        self.interactions.apply(self._init_weights)
        self.apply(self._init_attn_weights)
        self.gene_encoder.apply(self._init_weights)
        self.final_project.apply(self._init_weights)
        self.final_norm.apply(self._init_weights)
        # initialize gene positional embedding weights
        nn.init.trunc_normal_(self.gene_pe.data, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_attn_weights(self, m):
        if isinstance(m, CrossAttentionLayer):
            m._reset_parameters()
        if isinstance(m, SelfAttentionLayer):
            m._reset_parameters()

    def forward(
        self,
        x,
        coords,
        genes,
        task_token=None,
        attn_mask=None,
        multiway_split_position=None,
        incremental_state=None,
        **kwargs
    ):
        """
        The forward pass of the model

        Arguments:
        ----------
        x: torch.Tensor
            The input tile embeddings, of shape [N, L, D]
        coords: torch.Tensor
            The coordinates of the patches, of shape [N, L, 2]
        genes: torch.Tensor
            The genomic expression information of shape [N, L_gene]
        task_token: torch.Tensor
            should indicate task id in form of 1xN_tasks tensor
        """

        # embed patches
        x = self.patch_embed(x)

        # get pos indices
        pos = self.coords_to_pos(coords)  # [N, L]

        x = x + self.pos_embed[:, pos, :].squeeze(0)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x, encoder_embedding, encoder_padding_mask, rel_pos_bias = (
            self.encoder.prepare_forward(src_tokens=None, token_embeddings=x)
        )
        layer_configs = {
            "rel_pos": rel_pos_bias,
            "encoder_padding_mask": (
                encoder_padding_mask if incremental_state is None else None
            ),
            "attn_mask": attn_mask,
            "multiway_split_position": multiway_split_position,
        }

        # Gene encoder
        gene_embedding = self.gene_encoder(genes)

        if self.prompt_agg == "cls":
            gene_embedding = torch.cat((self.gene_cls, gene_embedding), dim=1)

        # append task prompts to gene encoder
        if self.is_multi:
            task_token = task_token.unsqueeze(0)  # 1xN_tasks
            task_cls = self.task_weight(task_token).unsqueeze(0)
            gene_embedding = torch.cat((task_cls, gene_embedding), dim=1)

        ############# BLOCK stuff ##############
        if self.interaction_indexes[0][0] != 0:
            for idx, blk in enumerate(
                self.encoder.layers[0 : self.interaction_indexes[0][0]]
            ):
                x, _ = blk(
                    x,
                    incremental_state=(
                        incremental_state[idx]
                        if incremental_state is not None
                        else None
                    ),
                    **layer_configs
                )

        # Interaction
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
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            # prompts selfattention
            # gene_embedding = self.prompt_selfattention[i](gene_embedding, gene_pe)
            gene_embedding = self.prompt_selfattention[i](gene_embedding, self.gene_pe)
            x, gene_embedding, cls = layer(
                x,
                gene_embedding,
                cls,
                self.encoder.layers[indexes[0] : indexes[-1] + 1],
                incremental_state,
                layer_configs,
                self.gene_pe,
            )
        #########################################
        if self.global_pool:
            img_outcome = x.mean(dim=1).unsqueeze(0)  # global average pooling
        else:
            img_outcome = cls

        if self.add_prompt_feature:
            if self.prompt_agg == "cls":
                task_outcome, gene_outcome, gene_embedding = (
                    gene_embedding[:, 0 : int(self.is_multi), :],
                    gene_embedding[:, int(self.is_multi) : int(self.is_multi) + 1, :],
                    gene_embedding[:, int(self.is_multi) + 1 :, :],
                )
            elif self.prompt_agg == "avg":
                task_outcome, gene_outcome = (
                    gene_embedding[:, 0 : int(self.is_multi), :],
                    gene_embedding[:, int(self.is_multi) :, :].mean(dim=1).unsqueeze(1),
                )
            else:
                raise NotImplementedError
            if self.token_agg == "sum":
                if self.is_multi:
                    outcome = img_outcome + gene_outcome + task_outcome
                    # outcome = task_outcome
                else:
                    outcome = img_outcome + gene_outcome
            elif self.token_agg == "cat":
                if self.is_multi:
                    outcome = torch.cat(
                        (img_outcome, task_outcome, gene_outcome), dim=-1
                    )
                else:
                    outcome = torch.cat((img_outcome, gene_outcome), dim=-1)
            else:
                raise NotImplementedError
        outcome = self.final_norm(outcome)
        outcome = self.final_project(
            outcome.squeeze(1)
        )  # 1x1xembed_dim -> 1xoutput_dim
        return outcome


@Aggregator.register("longnetvit_gene_clinical_adapter")
class LongNetGeneSimpleClinicalAdapter(LongNetGeneAdapter):
    """
    LongNet ViT Adapter with Gene Adapters and clinical priors
    """

    def __init__(
        self,
        num_heads=12,
        gene_group_defination={},
        geneclass_name="gene_mixer_group",
        output_dim=256,
        init_values=0.0,
        interaction_indexes=None,
        with_cffn=True,
        cffn_ratio=0.25,
        add_prompt_feature=True,
        use_extra_extractor=True,
        freeze_vit=True,
        with_cp=False,
        use_prompt_sa=False,
        prompt_dropout=0.0,
        prompt_agg="cls",
        token_agg="cat",
        pretrained=True,
        multi_task=1,
        clinfeat_dim=5,
        **kwargs
    ):
        """
        Parameters:
            interaction_indexes: List[List[int,int]], layers at which injectors and extractors would interact
            with_cffn: Set true to enable compression of feature space in attention layers
            cffn_ratio: Ratio of compression
            add_prompt_feature: Set true to add prompt feature to imaging feature
            use_extra_extractor: Set true to use extra extractors at the end of extractor step
            with_cp: Set true for checkpointing, see checkpoint for more details
            use_prompt_sa: Set true for using self attention
            prompt_dropout: dropout rate to be used in self attention of additional prompts
            prompt_agg: Choose out of cls/avg for getting final feature vector from gene tokens step
            token_agg: Choose out of cat/sum for how to combine final feature vectors from different modalities
            pretrained: Set true for loading the aggregator weights
            weights_location: Location for weights of the aggregator
            multitask: int, number of different embeddings for different tasks to calculate given same input
        """
        LongNetViT.__init__(self, **kwargs)
        # Load parameters after init
        self.load_slide_encoder(
            pretrained=pretrained, weights_location=GIGAPATH_WEIGHT_LOC
        )
        if freeze_vit:
            for name, param in self.named_parameters():
                param.requires_grad = False

        self.mode = "feature"
        self.num_block = self.depth
        self.interaction_indexes = interaction_indexes
        self.add_prompt_feature = add_prompt_feature
        self.prompt_agg = prompt_agg
        self.token_agg = token_agg
        self.is_multi = multi_task > 1
        embed_dim = self.embed_dim

        block_fn = InteractionBlockWithCls_LongNetViT

        self.interactions = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    init_values=init_values,
                    drop_path=self.drop_path_rate,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    extra_extractor=(
                        (True if i == len(interaction_indexes) - 1 else False)
                        and use_extra_extractor
                    ),
                    with_cp=with_cp,
                )
                for i in range(len(interaction_indexes))
            ]
        )

        self.prompt_selfattention = nn.Sequential(
            Identity_mod(),
            *[
                (
                    SelfAttentionLayer(
                        d_model=embed_dim,
                        nheads=num_heads,
                        dropout=prompt_dropout,
                        normalize_before=True,
                        with_cffn=with_cffn,
                        cffn_ratio=cffn_ratio,
                    )
                    if use_prompt_sa
                    else Identity_mod()
                )
                for _ in range(1, len(interaction_indexes))
            ]
        )

        num_gene_groups = len(gene_group_defination)
        args = argparse.Namespace(geneclass_name=geneclass_name, num_classes=2)
        print("Genomic model configurations: {}".format(set_genomic_config(args)))
        self.gene_encoder = GeneBaseClass.create(
            subclass_name=geneclass_name,
            **set_genomic_config(args),
            output_dim=embed_dim,
            mode="feature",
            group_sizes=gene_group_defination,
            n_groups=num_gene_groups
        )

        # to account for any changes based on cls token
        num_gene_groups = self.gene_encoder.n_groups
        # adding cls token
        if self.prompt_agg == "cls":
            self.gene_cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
            num_gene_groups = num_gene_groups + 1
            nn.init.trunc_normal_(self.gene_cls.data, std=0.02)

        # gene positional embeddings, add 1 for prompt token and 1 for clinical token
        self.gene_pe = nn.Parameter(
            torch.zeros(num_gene_groups + int(self.is_multi) + 1, embed_dim)
        )

        # multiple tasks
        # Converts NUM_TASKS tasks into promptable input
        if self.is_multi:
            self.task_weight = nn.Sequential(
                *[nn.Linear(multi_task, embed_dim), nn.LayerNorm(embed_dim)]
            )
            self.task_weight.apply(self._init_weights)

        self.clinical_mlp = nn.Sequential(
            nn.Linear(clinfeat_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.clinical_mlp.apply(self._init_weights)

        if self.token_agg == "cat":
            self.final_norm = nn.LayerNorm((3 + int(self.is_multi)) * embed_dim)
            self.final_project = nn.Linear(
                (3 + int(self.is_multi)) * embed_dim, output_dim
            )
        elif self.token_agg == "sum":
            self.final_norm = nn.LayerNorm(embed_dim)
            self.final_project = nn.Linear(embed_dim, output_dim)
        else:
            raise NotImplementedError

        # Initializations
        self.interactions.apply(self._init_weights)
        self.apply(self._init_attn_weights)
        self.gene_encoder.apply(self._init_weights)
        self.final_project.apply(self._init_weights)
        self.final_norm.apply(self._init_weights)
        # initialize gene positional embedding weights
        nn.init.trunc_normal_(self.gene_pe.data, std=0.02)

    def forward(
        self,
        x,
        coords,
        genes,
        clinical,
        task_token=None,
        attn_mask=None,
        multiway_split_position=None,
        incremental_state=None,
    ):
        """
        The forward pass of the model

        Arguments:
        ----------
        x: torch.Tensor
            The input tile embeddings, of shape [N, L, D]
        coords: torch.Tensor
            The coordinates of the patches, of shape [N, L, 2]
        genes: torch.Tensor
            The genomic expression information of shape [N, L_gene]
        task_token: torch.Tensor
            should indicate task id in form of 1xN_tasks tensor
        Rest of the args are particular to longvit
        """

        # embed patches
        x = self.patch_embed(x)

        # get pos indices
        pos = self.coords_to_pos(coords)  # [N, L]

        x = x + self.pos_embed[:, pos, :].squeeze(0)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x, encoder_embedding, encoder_padding_mask, rel_pos_bias = (
            self.encoder.prepare_forward(src_tokens=None, token_embeddings=x)
        )
        layer_configs = {
            "rel_pos": rel_pos_bias,
            "encoder_padding_mask": (
                encoder_padding_mask if incremental_state is None else None
            ),
            "attn_mask": attn_mask,
            "multiway_split_position": multiway_split_position,
        }

        # Gene encoder
        gene_embedding = self.gene_encoder(genes)

        if self.prompt_agg == "cls":
            gene_embedding = torch.cat((self.gene_cls, gene_embedding), dim=1)

        # append task prompts to gene encoder
        if self.is_multi:
            task_token = task_token.unsqueeze(0)  # 1xN_tasks
            task_cls = self.task_weight(task_token).unsqueeze(0)
            gene_embedding = torch.cat((task_cls, gene_embedding), dim=1)

        # clinical embedding
        clin_embedding = self.clinical_mlp(clinical).unsqueeze(0)
        gene_embedding = torch.cat((clin_embedding, gene_embedding), dim=1)

        ############# BLOCK stuff ##############
        if self.interaction_indexes[0][0] != 0:
            for idx, blk in enumerate(
                self.encoder.layers[0 : self.interaction_indexes[0][0]]
            ):
                x, _ = blk(
                    x,
                    incremental_state=(
                        incremental_state[idx]
                        if incremental_state is not None
                        else None
                    ),
                    **layer_configs
                )

        # Interaction
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
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            # prompts selfattention
            # gene_embedding = self.prompt_selfattention[i](gene_embedding, gene_pe)
            gene_embedding = self.prompt_selfattention[i](gene_embedding, self.gene_pe)
            x, gene_embedding, cls = layer(
                x,
                gene_embedding,
                cls,
                self.encoder.layers[indexes[0] : indexes[-1] + 1],
                incremental_state,
                layer_configs,
                self.gene_pe,
            )
        #########################################
        if self.global_pool:
            img_outcome = x.mean(dim=1).unsqueeze(0)  # global average pooling
        else:
            img_outcome = cls

        if self.add_prompt_feature:
            if self.prompt_agg == "cls":
                clinical_outcome, task_outcome, gene_outcome, gene_embedding = (
                    gene_embedding[:, 0:1, :],
                    gene_embedding[:, 1 : 1 + int(self.is_multi), :],
                    gene_embedding[
                        :, 1 + int(self.is_multi) : 2 + int(self.is_multi), :
                    ],
                    gene_embedding[:, int(self.is_multi) + 2 :, :],
                )
            elif self.prompt_agg == "avg":
                clinical_outcome, task_outcome, gene_outcome = (
                    gene_embedding[:, 0:1, :],
                    gene_embedding[:, 1 : 1 + int(self.is_multi), :],
                    gene_embedding[:, 1 + int(self.is_multi) :, :]
                    .mean(dim=1)
                    .unsqueeze(1),
                )
            else:
                raise NotImplementedError
            if self.token_agg == "sum":
                if self.is_multi:
                    outcome = (
                        img_outcome + gene_outcome + task_outcome + clinical_outcome
                    )
                    # outcome = task_outcome
                else:
                    outcome = img_outcome + gene_outcome + clinical_outcome
            elif self.token_agg == "cat":
                if self.is_multi:
                    outcome = torch.cat(
                        (img_outcome, task_outcome, gene_outcome, clinical_outcome),
                        dim=-1,
                    )
                else:
                    outcome = torch.cat(
                        (img_outcome, gene_outcome, clinical_outcome), dim=-1
                    )
            else:
                raise NotImplementedError
        outcome = self.final_norm(outcome)
        outcome = self.final_project(
            outcome.squeeze(1)
        )  # 1x1xembed_dim -> 1xoutput_dim
        return outcome
