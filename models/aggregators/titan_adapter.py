"""
Complete pipeline for ModalTune with TITAN slide encoder integrated with ModalAdapters
"""

import math
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
from safetensors import safe_open

from utils.constants import TITAN_CODE_PATH, TITAN_SNAPSHOT_ID

sys.path.append(TITAN_CODE_PATH)

from ..vitadapter.adapter_modules import (
    InteractionBlockWithCls_TITAN,
    SelfAttentionLayer,
    CrossAttentionLayer,
    Identity_mod,
)
from .aggregators import Aggregator
from ..genomic_utils.gene_baseclass import GeneBaseClass

# Dynamically import using the SNAPSHOT_ID
snapshot_module = __import__(
    f"{TITAN_SNAPSHOT_ID}.vision_transformer", fromlist=["VisionTransformer"]
)
VisionTransformer = snapshot_module.VisionTransformer
config_module = __import__(
    f"{TITAN_SNAPSHOT_ID}.configuration_titan", fromlist=["TitanConfig"]
)
TitanConfig = config_module.TitanConfig

from model_configs.other_configs import set_genomic_config


@Aggregator.register("titan_gene_adapter")
class TITANGeneAdapter(VisionTransformer, Aggregator):
    """
    TITAN - ViT Adapter with modal Adapters and task prompt
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
        use_prompt_sa=True,
        prompt_dropout=0.0,
        prompt_agg="avg",
        token_agg="sum",
        pretrained=True,
        multi_task=1,
        drop_path_rate=0.1,
        **kwargs,
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
            drop_path_rate: float, drop path rate for stochastic depth
        """
        self.vision_cfg = TitanConfig().vision_config
        # from build_vision_tower
        VisionTransformer.__init__(
            self,
            grid_size=self.vision_cfg.grid_size,
            global_pool=self.vision_cfg.global_pool,
            embed_dim=self.vision_cfg.embed_dim,
            depth=self.vision_cfg.depth,
            num_heads=self.vision_cfg.num_heads,
            mlp_ratio=self.vision_cfg.mlp_ratio,
            qkv_bias=self.vision_cfg.qkv_bias,
            mlp_patch_embed_dim=self.vision_cfg.mlp_patch_embed_dim,
            pos_encode_type=self.vision_cfg.pos_encode_type,
            attentional_pool=self.vision_cfg.attentional_pool,
            attn_pooler_queries=self.vision_cfg.attn_pooler_queries,
            attn_pooler_heads=self.vision_cfg.attn_pooler_heads,
        )
        # Load parameters after init
        self.load_slide_encoder(pretrained=pretrained)
        if freeze_vit:
            for name, param in self.named_parameters():
                param.requires_grad = False

        self.mode = "feature"
        self.num_block = self.vision_cfg.depth
        self.interaction_indexes = interaction_indexes
        self.add_prompt_feature = add_prompt_feature
        self.prompt_agg = prompt_agg
        self.token_agg = token_agg
        self.is_multi = multi_task > 1
        embed_dim = self.vision_cfg.embed_dim

        block_fn = InteractionBlockWithCls_TITAN

        self.interactions = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    init_values=init_values,
                    drop_path=drop_path_rate,
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
            ],
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
            n_groups=num_gene_groups,
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

    def load_slide_encoder(self, pretrained=True):
        if pretrained:
            tensors = {}
            with safe_open(
                Path(TITAN_CODE_PATH) / TITAN_SNAPSHOT_ID / "model.safetensors",
                framework="pt",
                device=0,
            ) as f:
                for k in f.keys():
                    if "vision_encoder" in k:
                        tensors[k.split("vision_encoder.")[1]] = f.get_tensor(k)
            missing_keys, unexpected_keys = self.load_state_dict(tensors)
            print(f"Missing keys: {missing_keys} ")
            print(f"Unexpected keys: {unexpected_keys} ")

    def prepare_forward_features(self, x, coords, mask=None, bg_mask=None):
        B, nc, w, h = x.shape
        ## flatten x before pass into mlp patch embed
        x = x.flatten(2, 3).transpose(1, 2)
        if self.pos_encode_type == "alibi":
            if w * h == 36 and B != 1:
                if not self.local_alibi_status:
                    self.prepare_tensor(x, "local", "alibi")
                attn_bias = self.local_alibi
            elif w * h == 196 and B != 1:
                if not self.global_alibi_status:
                    self.prepare_tensor(x, "global", "alibi")
                attn_bias = self.global_alibi
            else:
                attn_bias = (
                    self.get_alibi(w, h, bg_mask) if B == 1 else self.get_alibi(w, h)
                )
                attn_bias = (
                    attn_bias.repeat(x.shape[0], 1, 1, 1).type(x.dtype).to(x.device)
                )
        else:
            attn_bias = None

        if self.masked_im_modeling:
            assert mask is not None
            x = self.patch_embed(x)
            x = self.mask_model(x, mask)
        else:
            x = self.patch_embed(x)

        x = self._pos_embed(x, coords, w, h)
        x = self.norm_pre(x)

        # mask background tokens when evaluating (batch size = 1)
        if bg_mask is not None and B == 1:
            bg_mask = torch.cat(
                (
                    torch.ones((1, 1), dtype=torch.bool, device=x.device),
                    bg_mask.view(1, -1),
                ),
                dim=1,
            )
            x = x[bg_mask].unsqueeze(0)

        return x, attn_bias, bg_mask

    @staticmethod
    def preprocess_features(
        features: torch.Tensor, coords: torch.Tensor, patch_size_lv0: int
    ):
        # Remove extra dimensions
        features = features.squeeze(0) if features.dim() == 3 else features
        coords = coords.squeeze(0) if coords.dim() == 3 else coords

        # Offset and normalize coordinates
        offset = coords.min(dim=0).values
        grid_coords = torch.floor_divide(coords - offset, patch_size_lv0)

        # Compute grid size
        grid_offset = grid_coords.min(dim=0).values
        grid_coords = grid_coords - grid_offset
        _H, _W = grid_coords.max(dim=0).values + 1

        # Create feature and coordinate grids
        feature_grid = torch.zeros((_H, _W, features.size(-1)), device=features.device)
        coords_grid = torch.zeros((_H, _W, 2), dtype=torch.int64, device=coords.device)

        # Use scatter for more efficient placement
        indices = grid_coords[:, 0] * _W + grid_coords[:, 1]
        feature_grid.view(-1, features.size(-1)).index_add_(0, indices, features)
        coords_grid.view(-1, 2).index_add_(0, indices, coords)

        # Permute grids
        feature_grid = feature_grid.permute(2, 0, 1)
        coords_grid = coords_grid.permute(2, 0, 1)

        # Background mask
        bg_mask = torch.any(feature_grid != 0, dim=0)
        return feature_grid.unsqueeze(0), coords_grid.unsqueeze(0), bg_mask.unsqueeze(0)

    def forward(
        self,
        x,
        coords,
        genes,
        task_token=None,
        patch_size_lv0=1024,
        **kwargs,
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
        x, coords, bg_mask = self.preprocess_features(x, coords, patch_size_lv0)
        x, attn_bias, bg_mask = self.prepare_forward_features(
            x, coords=coords, mask=None, bg_mask=bg_mask
        )

        if self.interaction_indexes[0][0] != 0:
            for idx, blk in enumerate(
                self.blocks.modules_list[0 : self.interaction_indexes[0][0]]
            ):
                x = blk(x, attn_bias, bg_mask)

        # Gene encoder
        gene_embedding = self.gene_encoder(genes)

        if self.prompt_agg == "cls":
            gene_embedding = torch.cat((self.gene_cls, gene_embedding), dim=1)

        # append task prompts to gene encoder
        if self.is_multi:
            task_token = task_token.unsqueeze(0)  # 1xN_tasks
            task_cls = self.task_weight(task_token).unsqueeze(0)
            gene_embedding = torch.cat((task_cls, gene_embedding), dim=1)

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
            gene_embedding = self.prompt_selfattention[i](gene_embedding, self.gene_pe)
            x, gene_embedding, cls = layer(
                x,
                gene_embedding,
                cls,
                self.blocks.modules_list[indexes[0] : indexes[-1] + 1],
                attn_bias,
                bg_mask,
                self.gene_pe,
            )

        x = torch.cat((cls, x), dim=1)
        x = self.norm(x)
        img_outcome, tokens = self.forward_attn_pool(x, bg_mask=bg_mask)
        img_outcome = img_outcome.unsqueeze(0)

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


@Aggregator.register("titan_gene_clinical_adapter")
class TITANGeneSimpleClinicalAdapter(TITANGeneAdapter):
    """
    TITAN - ViT Adapter with modal Adapters and task prompt
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
        use_prompt_sa=True,
        prompt_dropout=0.0,
        prompt_agg="avg",
        token_agg="sum",
        pretrained=True,
        multi_task=1,
        drop_path_rate=0.1,
        **kwargs,
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
        self.vision_cfg = TitanConfig().vision_config
        # from build_vision_tower
        VisionTransformer.__init__(
            self,
            grid_size=self.vision_cfg.grid_size,
            global_pool=self.vision_cfg.global_pool,
            embed_dim=self.vision_cfg.embed_dim,
            depth=self.vision_cfg.depth,
            num_heads=self.vision_cfg.num_heads,
            mlp_ratio=self.vision_cfg.mlp_ratio,
            qkv_bias=self.vision_cfg.qkv_bias,
            mlp_patch_embed_dim=self.vision_cfg.mlp_patch_embed_dim,
            pos_encode_type=self.vision_cfg.pos_encode_type,
            attentional_pool=self.vision_cfg.attentional_pool,
            attn_pooler_queries=self.vision_cfg.attn_pooler_queries,
            attn_pooler_heads=self.vision_cfg.attn_pooler_heads,
        )
        # Load parameters after init
        self.load_slide_encoder(pretrained=pretrained)
        if freeze_vit:
            for name, param in self.named_parameters():
                param.requires_grad = False

        self.mode = "feature"
        self.num_block = self.vision_cfg.depth
        self.interaction_indexes = interaction_indexes
        self.add_prompt_feature = add_prompt_feature
        self.prompt_agg = prompt_agg
        self.token_agg = token_agg
        self.is_multi = multi_task > 1
        embed_dim = self.vision_cfg.embed_dim

        block_fn = InteractionBlockWithCls_TITAN

        self.interactions = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    init_values=init_values,
                    drop_path=drop_path_rate,
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
            ],
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
            n_groups=num_gene_groups,
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
            nn.Linear(6, embed_dim // 2),
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
        patch_size_lv0=1024,
        **kwargs,
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
        patch_size_lv0: int
            patch size at level 0 (titan specific argument)
        """
        x, coords, bg_mask = self.preprocess_features(x, coords, patch_size_lv0)
        x, attn_bias, bg_mask = self.prepare_forward_features(
            x, coords=coords, mask=None, bg_mask=bg_mask
        )

        if self.interaction_indexes[0][0] != 0:
            for idx, blk in enumerate(
                self.blocks.modules_list[0 : self.interaction_indexes[0][0]]
            ):
                x = blk(x, attn_bias, bg_mask)

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
            gene_embedding = self.prompt_selfattention[i](gene_embedding, self.gene_pe)
            x, gene_embedding, cls = layer(
                x,
                gene_embedding,
                cls,
                self.blocks.modules_list[indexes[0] : indexes[-1] + 1],
                attn_bias,
                bg_mask,
                self.gene_pe,
            )

        x = torch.cat((cls, x), dim=1)
        x = self.norm(x)
        img_outcome, tokens = self.forward_attn_pool(x, bg_mask=bg_mask)
        img_outcome = img_outcome.unsqueeze(0)

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
