"""
Convert clinical data to dictionary format for training
{CASE_ID: [CLIN FEAT 1, CLIN FEAT 2.....,CLIN FEAT N]}
"""

import argparse
import pandas as pd
from pathlib import Path
from sklearn import preprocessing
import json
import torch


def prepare_clinical_features(df, path):
    """
    Prepare clinical features by cleaning and encoding them, then save as a dictionary.
    """
    df_analyze = df.copy()
    df_analyze.loc[
        ~df_analyze["ajcc_pathologic_stage"].isna(), "ajcc_pathologic_stage"
    ] = df.loc[
        ~df["ajcc_pathologic_stage"].isna(), "ajcc_pathologic_stage"
    ].apply(
        lambda x: x.replace("A", "")
        .replace("B", "")
        .replace("b", "")
        .replace("C", "")
        .replace("c", "")
        .replace("D", "")
        .replace("d", "")
        .replace(" (i+)", "")
        .replace(" (i-)", "")
        .replace("m", "")
        .replace("i", "")
    )
    df_analyze.loc[
        ~df_analyze["ajcc_pathologic_t"].isna(), "ajcc_pathologic_t"
    ] = df.loc[
        ~df["ajcc_pathologic_t"].isna(), "ajcc_pathologic_t"
    ].apply(
        lambda x: x.replace("A", "")
        .replace("a", "")
        .replace("B", "")
        .replace("b", "")
        .replace("C", "")
        .replace("c", "")
        .replace("D", "")
        .replace("d", "")
        .replace(" (i+)", "")
        .replace(" (i-)", "")
        .replace("m", "")
        .replace("is", "0")
        .replace("i", "")
    )
    df_analyze.loc[
        ~df_analyze["ajcc_pathologic_n"].isna(), "ajcc_pathologic_n"
    ] = df.loc[
        ~df["ajcc_pathologic_n"].isna(), "ajcc_pathologic_n"
    ].apply(
        lambda x: x.replace("A", "")
        .replace("a", "")
        .replace("B", "")
        .replace("b", "")
        .replace("C", "")
        .replace("c", "")
        .replace("D", "")
        .replace("d", "")
        .replace(" (i+)", "")
        .replace(" (i-)", "")
        .replace(" (mol+)", "")
        .replace("m", "")
        .replace("i", "")
    )
    df_analyze.loc[
        ~df_analyze["ajcc_pathologic_m"].isna(), "ajcc_pathologic_m"
    ] = df.loc[
        ~df["ajcc_pathologic_m"].isna(), "ajcc_pathologic_m"
    ].apply(
        lambda x: x.replace("A", "")
        .replace("a", "")
        .replace("B", "")
        .replace("b", "")
        .replace("C", "")
        .replace("c", "")
        .replace("D", "")
        .replace("d", "")
        .replace(" (i+)", "")
        .replace(" (i-)", "")
        .replace("m", "")
        .replace("i", "")
    )
    df_analyze = df_analyze.fillna("Nan")
    df_analyze["age_at_index"] = (
        df_analyze["age_at_index"].astype(float)
        / df_analyze["age_at_index"].astype(float).max()
    )

    df_analyze["ajcc_pathologic_stage"] = preprocessing.LabelEncoder().fit_transform(
        df_analyze["ajcc_pathologic_stage"]
    )
    df_analyze["ajcc_pathologic_t"] = preprocessing.LabelEncoder().fit_transform(
        df_analyze["ajcc_pathologic_t"]
    )
    df_analyze["ajcc_pathologic_n"] = preprocessing.LabelEncoder().fit_transform(
        df_analyze["ajcc_pathologic_n"]
    )
    df_analyze["ajcc_pathologic_m"] = preprocessing.LabelEncoder().fit_transform(
        df_analyze["ajcc_pathologic_m"]
    )

    d = {}
    for i in range(len(df_analyze)):
        d[df_analyze.iloc[i, 0]] = torch.tensor(
            df_analyze.iloc[i, 1:].values.astype(float)
        )
    torch.save(d, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onco_code", default="rcc", type=str, help="data code")
    parser.add_argument(
        "--output_dir",
        default="./data",
        type=str,
        help="directory for storing processed files",
    )
    args = parser.parse_args()
    onco_code = args.onco_code.lower()
    output_dir = Path(args.output_dir)
    ROOT_DIR = Path(__file__).resolve().parent.parent
    # load jsons
    with open(
        str(
            ROOT_DIR
            / f"dataset/json_splits/tcga_{onco_code}/train_{onco_code}_cls_feat.json"
        ),
        "r",
    ) as file:
        train_datalist = json.load(file)["data"]
        train_df = pd.DataFrame(train_datalist)
    with open(
        str(
            ROOT_DIR
            / f"dataset/json_splits/tcga_{onco_code}/val_{onco_code}_cls_feat.json"
        ),
        "r",
    ) as file:
        val_datalist = json.load(file)["data"]
        val_df = pd.DataFrame(val_datalist)
    with open(
        str(
            ROOT_DIR
            / f"dataset/json_splits/tcga_{onco_code}/test_{onco_code}_cls_feat.json"
        ),
        "r",
    ) as file:
        test_datalist = json.load(file)["data"]
        test_df = pd.DataFrame(test_datalist)
    df = pd.concat((train_df, val_df, test_df), axis=0)
    df.drop_duplicates(subset="case_id", inplace=True, ignore_index=True)
    # If additional information is there can be added here
    ## <- ADD MORE CLINICAL FEATURES ->
    df = df[
        [
            "case_id",
            "age_at_index",
            "ajcc_pathologic_stage",
            "ajcc_pathologic_t",
            "ajcc_pathologic_n",
            "ajcc_pathologic_m",
        ]
    ]
    prepare_clinical_features(df, output_dir / f"simple_clinical_dict_{onco_code}.pt")
