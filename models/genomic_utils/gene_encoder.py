"""
Gene Encoder for converting N x 20530 gene encoding into N x N_groups x 1536
The mixer idea is taken from 
Code adapted from https://github.com/lucidrains/mlp-mixer-pytorch/blob/main/mlp_mixer_pytorch/mlp_mixer_pytorch.py
"""

from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class GeneBaseClass(nn.Module):
    """
    To register the subclasses based on the name
    To be used as
    @GeneBaseClass.register("some_name")
    class SomeClass(GeneBaseClass)

    To build that class, one can use
    GeneBaseClass.create("some_name")
    """

    subclasses = {}

    def __init__(self, mode="classifier") -> None:
        super().__init__()
        self.mode = mode

    @classmethod
    def register(cls, subclass_name: str):
        def decorator(subclass: Any):
            subclass.subclasses[subclass_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, subclass_name: str, **params):
        if subclass_name not in cls.subclasses:
            raise ValueError("Unknown subclass name {}".format(subclass_name))
        print("-" * 50)
        print(
            f"For class: {cls.__name__}, Selected subclass: ({subclass_name}):{cls.subclasses[subclass_name]}"
        )
        print("-" * 50)

        return cls.subclasses[subclass_name](**params)

    def genes_encode(self, *args, **kwargs):
        """
        Encodes the genomics information into features
        Expected output -> [1 x output_dim] or [1 x n_groups x output_dim]
        """
        raise NotImplemented

    def forward(self, x):
        """
        For performing prediction by returning output of form -> [1 x n_classes/output_dim] based on the selected mode
        """
        x = self.gene_encode(x)
        gene_outcome, gene_embedding = (x[:, :1, :], x[:, 1:, :])
        gene_outcome = self.final_norm(gene_outcome).squeeze()

        return self.return_logits(gene_outcome)

    def return_logits(self, h):
        """
        features: 1 x n_dim
        """
        if self.mode == "feature":
            return h
        logits = self.classifier(h).unsqueeze(0)  # logits needs to be a [1 x 4] vector
        if self.mode == "classifier":
            return logits.squeeze()
        elif self.mode == "survival":
            Y_hat = torch.topk(logits, 1, dim=1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat
        else:
            raise NotImplementedError


@GeneBaseClass.register("gene_mixer_group")
class GeneEncoder_Group(GeneBaseClass):
    def __init__(
        self,
        output_dim: int,
        latent_dim: int,
        group_sizes: dict,
        n_groups: int = 64,
        depth: int = 5,
        cls_token: bool = False,
        expansion_groups=4,
        expansion_dim=0.5,
        dropout: float = 0.25,
        n_classes: int = 2,
        mode: str = "classifier",
        final_groups: int = 8,
        **kwargs,
    ):
        super().__init__(mode=mode)

        # Handling groups of genes
        hidden = [latent_dim, latent_dim]
        gene_networks = []
        for input_genes in group_sizes.values():
            fc_omic = [
                self.SNN_Block(dim1=len(input_genes), dim2=hidden[0], dropout=dropout)
            ]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(
                    self.SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=dropout)
                )
            gene_networks.append(nn.Sequential(*fc_omic))
        self.gene_networks = nn.ModuleList(gene_networks)

        # Mixers
        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        self.is_cls_token = cls_token

        if self.is_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
            n_groups = n_groups + 1

        self.mlp_mixer = nn.Sequential(
            *[
                nn.Sequential(
                    PreNormResidual(
                        latent_dim,
                        self.FeedForward(
                            n_groups, expansion_groups, dropout, chan_first
                        ),
                    ),
                    PreNormResidual(
                        latent_dim,
                        self.FeedForward(latent_dim, expansion_dim, dropout, chan_last),
                    ),
                )
                for _ in range(depth)
            ],
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, output_dim),
        )

        if self.mode != "feature":
            self.final_norm = nn.LayerNorm(output_dim)
            self.classifier = nn.Linear(output_dim, n_classes)

        # Testing smaller number of tokens
        self.n_groups = final_groups
        self.pathway_compression = nn.Linear(n_groups, final_groups)

    @staticmethod
    def SNN_Block(dim1, dim2, dropout=0.25):
        r"""
        Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

        args:
            dim1 (int): Dimension of input features
            dim2 (int): Dimension of output features
            dropout (float): Dropout rate
        """
        return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(dropout),
        )

    @staticmethod
    def FeedForward(dim, expansion_factor=4, dropout=0.0, dense=nn.Linear):
        inner_dim = int(dim * expansion_factor)
        return nn.Sequential(
            dense(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            dense(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def gene_encode(self, x):
        """
        Processes input of form
            x: Dict[List] = {0:[],1:[]....}
        Returns:
            output: n_groups x output_dim
        """
        x_in = []
        for i in range(len(x)):
            x_in.append(self.gene_networks[i](x[i]))
        x = torch.cat(x_in).unsqueeze(0)
        if self.is_cls_token:
            cls_token = self.cls_token.expand(
                x.shape[0], -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_token, x), dim=1)
        x = self.mlp_mixer(x)
        # Testing smaller number of tokens
        x = self.pathway_compression(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x

    def forward(self, x):
        x = self.gene_encode(x)
        if self.mode != "feature":
            gene_outcome = x.mean(dim=1)
            gene_outcome = self.final_norm(gene_outcome).squeeze()

            return self.return_logits(gene_outcome)
        else:
            return x
