"""
Remove those genes which are constant and format in code friendly way. Additionally processes to make
it consistent with the pathways defined in Survpath (https://github.com/mahmoodlab/SurvPath/tree/main)
Transcriptomics data downloaded from Xena database
"""

import argparse
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

from gene_thesaurus import GeneThesaurus

gt = GeneThesaurus(data_dir="/tmp")  # for consistent naming convention


def process_only_gene(pancancer_data_loc, rna_seq_loc):
    """
    Process the pancancer and specific cancer type gene expression data to filter out constant genes.
    Args:
        pancancer_data_loc: path to the pancancer gene expression data file
        rna_seq_loc: path to the specific cancer type gene expression data file
    """
    df_pancan = pd.read_csv(pancancer_data_loc, sep="\t")
    df_pancan = df_pancan.drop_duplicates(subset=["sample"])
    std_val = df_pancan.iloc[:, 1:].std(axis=1)
    # number of genes is 19648 where #213 of them are constant values. Removing constant gene values
    df_proc = df_pancan.loc[std_val > 0]

    xena_data = rna_seq_loc
    df_xena = pd.read_csv(xena_data, sep="\t")
    # gene id different way
    df_xena["sample"] = df_xena["sample"].apply(lambda x: x[2:] if x[:2] == "?|" else x)
    assert len(set(df_pancan["sample"]) - set(df_xena["sample"])) == 0

    # Select $num_genes_select gene ids from pancan
    df_xena_proc = df_xena.loc[df_xena["sample"].isin(list(df_proc["sample"]))]

    # further processing
    df_xena_proc = df_xena_proc.set_index("sample")
    df_xena_proc = df_xena_proc.T
    df_xena_proc = df_xena_proc.reset_index(drop=False, inplace=False)
    df_xena_proc.rename(columns={"index": "case_id"}, inplace=True)
    df_xena_proc["case_id"] = df_xena_proc["case_id"].apply(
        lambda x: ("-").join(x.split("-")[:-1])
    )
    df_xena_proc = df_xena_proc.drop_duplicates(subset=["case_id"])
    return df_xena_proc


def process_pathway(data, pathway_grouping_loc):
    """Process the pathway grouping file to map genes to pathways.
    Args:
        data: DataFrame containing gene expression data with gene symbols as columns
        pathway_grouping_loc: path to save the processed pathway grouping file
    """
    # load pathway association with genes taken from survpath
    pathways = pd.read_csv("../dataset/combine_comps.csv")
    gene_list = pathways["gene"].tolist()
    current_list = set(data.columns[1:].tolist())
    gene_list_trans = gt.translate_genes(
        gene_list, source="symbol", target="ensembl_id"
    )
    current_list_trans = gt.translate_genes(
        list(current_list), source="symbol", target="ensembl_id"
    )

    initial_matches = set(gene_list).intersection(set(current_list))
    gene_list_left = list(set(gene_list) - initial_matches)
    gene_list_left_trans = gt.translate_genes(
        gene_list_left, source="symbol", target="ensembl_id"
    )

    op_map_left = dict((v, k) for k, v in gene_list_left_trans.items())
    print("Warning... Unmatched genes from the given two input tables.")
    for genes in set(list(gene_list_left_trans.values())) - set(
        list(gene_list_left_trans.values())
    ).intersection(set(list(current_list_trans.values()))):
        print(f"{op_map_left[genes]}:{genes}")

    trans = {}
    for genes in initial_matches:
        trans[genes] = genes

    op_map_left = dict((v, k) for k, v in gene_list_left_trans.items())
    op_map_current = dict((v, k) for k, v in current_list_trans.items())
    for genes in set(list(gene_list_left_trans.values())).intersection(
        set(list(current_list_trans.values()))
    ):
        trans[op_map_left[genes]] = op_map_current[genes]

    pathways_new = pathways.loc[pathways["gene"].isin(list(trans.keys()))]
    pathways_new["gene"] = pathways_new["gene"].apply(lambda x: trans[x])
    # remove duplicate gene rows
    pathways_new = pathways_new.drop_duplicates(subset=["gene"])
    pathways_new.reset_index(drop=True, inplace=True)
    pathways_new.to_csv(pathway_grouping_loc, index=False)
    return pathways_new

def process_all(
    pancancer_data_loc, rna_seq_loc, pathway_grouping_loc, processed_data_loc
):
    """Process the raw gene expression data to filter genes and map to pathways.
    Args:
        pancancer_data_loc: path to the pancancer gene expression data file
        rna_seq_loc: path to the specific cancer type gene expression data file
        pathway_grouping_loc: path to save/load the processed pathway grouping file
        processed_data_loc: path to save the final processed gene expression data
    """
    # perform initial processing
    data = process_only_gene(pancancer_data_loc, rna_seq_loc)
    # get pathway
    if Path(pathway_grouping_loc).exists():
        print("Found groupings...")
        pathways_new = pd.read_csv(pathway_grouping_loc)
    else:
        print("Did not find groupings, creating...")
        pathways_new = process_pathway(data, pathway_grouping_loc)
    # Filter same set of genes
    print("Processing...")
    data = data[["case_id"] + pathways_new["gene"].to_list()]
    data.to_csv(processed_data_loc, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onco_code", default="brca", type=str, help="data code")
    parser.add_argument(
        "--raw_data_dir",
        default="./raw_data",
        type=str,
        help="directory containing of raw input files",
    )
    parser.add_argument(
        "--output_dir",
        default="./data",
        type=str,
        help="directory for storing processed files",
    )
    args = parser.parse_args()

    onco_code = "tcga_" + args.onco_code.lower()
    raw_data_dir = Path(args.raw_data_dir)
    output_dir = Path(args.output_dir)

    pancancer_data_loc = (
        raw_data_dir / "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena"
    )
    rna_seq_loc = raw_data_dir / f"{onco_code}_HiSeqV2_PANCAN"
    processed_data_loc = output_dir / f"{onco_code}_xena_clean_pathway.csv"
    pathway_grouping_loc = ROOT_DIR / "dataset/gene_pathway_processed_v2.csv"

    process_all(
        pancancer_data_loc, rna_seq_loc, pathway_grouping_loc, processed_data_loc
    )
