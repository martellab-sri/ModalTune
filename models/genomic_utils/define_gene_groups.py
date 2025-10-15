"""
Script for getting dictionary linking each gene to their corresponding pathways
Returns
{
    PATHWAY_i: [GENE_x, GENE_y, GENE_z],
    PATHWAY_j: [GENE_y, GENE_z],
    ...
}
"""

import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def pathway_gene_groups(group_definations: int = 0, *args, **kwargs):
    """
    Given dataframe of genes, gives the cluster id
    """
    if group_definations:
        pathway = pd.read_csv(ROOT_DIR / "dataset/gene_pathway_processed_v2.csv")
        gene_group_defination = {}
        for i, path in enumerate(pathway.columns[1:]):
            gene_group_defination[i] = pathway.loc[pathway[path] == 1, "gene"].to_list()
    else:
        gene_group_defination = {}

    return gene_group_defination
