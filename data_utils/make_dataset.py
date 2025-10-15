"""
Main script for pre-processing data and dividing into train/val/test splits
Saves the processed data in json format stored inside dataset folder
"""

import glob
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json


def cancer_specific_filter(df_filter, onco_code):
    """
    Filter and map primary diagnoses to classes based on cancer type.
    """
    df_filter["primary_diagnosis"] = df_filter["primary_diagnosis"].apply(
        lambda x: x.replace(", NOS", "")
    )
    df_filter["primary_class"] = -1
    if onco_code == "brca":
        class_dict = {"Infiltrating duct carcinoma": 0, "Lobular carcinoma": 1}
        for key, value in class_dict.items():
            df_filter.loc[df_filter["primary_diagnosis"] == key, "primary_class"] = (
                value
            )
    if onco_code == "gbmlgg":
        class_dict = {
            "Glioblastoma": 0,
            "Mixed glioma": 1,
            "Oligodendroglioma": 1,
            "Astrocytoma": 1,
            "Oligodendroglioma, anaplastic": 1,
            "Astrocytoma, anaplastic": 1,
        }
        for key, value in class_dict.items():
            df_filter.loc[df_filter["primary_diagnosis"] == key, "primary_class"] = (
                value
            )
    if onco_code == "nsclc":
        change_dict = {
            "Adenocarcinoma with mixed subtypes": "Adenocarcinoma",
            "Squamous cell carcinoma, keratinizing": "Squamous cell carcinoma",
            "Squamous cell carcinoma, large cell, nonkeratinizing": "Squamous cell carcinoma",
            "Bronchiolo-alveolar carcinoma, non-mucinous": "Bronchiolo-alveolar carcinoma",
            "Bronchio-alveolar carcinoma, mucinous": "Bronchiolo-alveolar carcinoma",
            "Bronchio-alveolar carcinoma": "Bronchiolo-alveolar carcinoma",
        }
        class_dict = {
            "Adenocarcinoma": 0,
            "Squamous cell carcinoma": 1,
        }
        df_filter.loc[
            df_filter["primary_diagnosis"].isin(change_dict.keys()), "primary_diagnosis"
        ] = (
            df_filter.loc[
                df_filter["primary_diagnosis"].isin(change_dict.keys()),
                "primary_diagnosis",
            ]
            .apply(lambda x: change_dict[x])
            .values
        )
        for key, value in class_dict.items():
            df_filter.loc[df_filter["primary_diagnosis"] == key, "primary_class"] = (
                value
            )
        df_filter["primary_diagnosis"] = df_filter["primary_diagnosis"].apply(
            lambda x: "Lung " + x
        )
    if onco_code == "coadread":
        change_dict = {
            "Colon Adenocarcinoma with mixed subtypes": "Colon Adenocarcinoma",
            "Rectal Adenocarcinoma with mixed subtypes": "Rectal Adenocarcinoma",
        }
        class_dict = {
            "Colon Adenocarcinoma": 0,
            "Rectal Adenocarcinoma": 1,
        }
        df_filter.loc[df_filter["project_id"] == "TCGA-COAD", "primary_diagnosis"] = (
            df_filter.loc[
                df_filter["project_id"] == "TCGA-COAD", "primary_diagnosis"
            ].apply(lambda x: "Colon " + x)
        )
        df_filter.loc[df_filter["project_id"] == "TCGA-READ", "primary_diagnosis"] = (
            df_filter.loc[
                df_filter["project_id"] == "TCGA-READ", "primary_diagnosis"
            ].apply(lambda x: "Rectal " + x)
        )
        df_filter.loc[
            df_filter["primary_diagnosis"].isin(change_dict.keys()), "primary_diagnosis"
        ] = (
            df_filter.loc[
                df_filter["primary_diagnosis"].isin(change_dict.keys()),
                "primary_diagnosis",
            ]
            .apply(lambda x: change_dict[x])
            .values
        )
        for key, value in class_dict.items():
            df_filter.loc[df_filter["primary_diagnosis"] == key, "primary_class"] = (
                value
            )
    if onco_code == "rcc":
        change_dict = {
            "Papillary adenocarcinoma": "Papillary renal cell carcinoma",  # kirp, acc to oncotype tree
            "Clear cell adenocarcinoma": "Renal clear cell carcinoma",  # present in kich, acc to oncotype tree
            "Renal cell carcinoma": "Renal clear cell carcinoma",
            "Renal cell carcinoma, chromophobe type": "Chromophobe renal cell carcinoma",
        }  # present in kich, acc to oncotype tree
        class_dict = {
            "Papillary renal cell carcinoma": 0,
            "Renal clear cell carcinoma": 1,
            "Chromophobe renal cell carcinoma": 2,
        }
        df_filter.loc[
            df_filter["primary_diagnosis"].isin(change_dict.keys()), "primary_diagnosis"
        ] = (
            df_filter.loc[
                df_filter["primary_diagnosis"].isin(change_dict.keys()),
                "primary_diagnosis",
            ]
            .apply(lambda x: change_dict[x])
            .values
        )
        for key, value in class_dict.items():
            df_filter.loc[df_filter["primary_diagnosis"] == key, "primary_class"] = (
                value
            )
    if onco_code == "ucec":
        change_dict = {
            "Endometrioid adenocarcinoma, secretory variant": "Endometrioid adenocarcinoma",
            "Papillary serous cystadenocarcinoma": "Serous cystadenocarcinoma",
            "Adenocarcinoma": "Endometrioid adenocarcinoma",
            "Serous surface papillary carcinoma": "Serous cystadenocarcinoma",
        }
        class_dict = {"Endometrioid adenocarcinoma": 0, "Serous cystadenocarcinoma": 1}
        df_filter.loc[
            df_filter["primary_diagnosis"].isin(change_dict.keys()), "primary_diagnosis"
        ] = (
            df_filter.loc[
                df_filter["primary_diagnosis"].isin(change_dict.keys()),
                "primary_diagnosis",
            ]
            .apply(lambda x: change_dict[x])
            .values
        )
        for key, value in class_dict.items():
            df_filter.loc[df_filter["primary_diagnosis"] == key, "primary_class"] = (
                value
            )
    if onco_code == "blca":
        change_dict = {
            "Papillary adenocarcinoma": "Papillary transitional cell carcinoma"
        }
        class_dict = {
            "Transitional cell carcinoma": 0,
            "Papillary transitional cell carcinoma": 1,
        }
        df_filter.loc[
            df_filter["primary_diagnosis"].isin(change_dict.keys()), "primary_diagnosis"
        ] = (
            df_filter.loc[
                df_filter["primary_diagnosis"].isin(change_dict.keys()),
                "primary_diagnosis",
            ]
            .apply(lambda x: change_dict[x])
            .values
        )
        for key, value in class_dict.items():
            df_filter.loc[df_filter["primary_diagnosis"] == key, "primary_class"] = (
                value
            )
    df_filter["primary_diagnosis"] = df_filter["primary_diagnosis"].apply(
        lambda x: x.lower()
    )
    return df_filter


def load_labelset(onco_code, data_path_list, label_path, labelset):
    """
    Load the clinical data and slide information for the specified cancer type.
    Here we assume that the clinical data is stored in a folder named TCGA-<ONCO_CODE>
    under the label_path directory.
    Args:
        onco_code: str, cancer type code (e.g., 'brca', 'ucec', 'blca')
        data_path_list: list of paths to the slide feature files
        label_path: path to the directory containing clinical data
        labelset: list of clinical features to be included without any NA values
    """
    # Process labelset
    df = pd.read_csv(f"{label_path}/clinical/clinical.tsv", sep="\t")

    # Merge with slide data
    slide_ids = pd.read_csv(f"{label_path}/biospecimen/slide.tsv", sep="\t")
    df = df.merge(
        slide_ids[["case_id", "slide_submitter_id"]],
        how="left",
        left_on="case_id",
        right_on="case_id",
        suffixes=(None, None),
    )
    df.replace("'--", np.NaN, inplace=True)
    # only consider available slides
    available_slides = [Path(temp).stem.split("_")[0] for temp in data_path_list]
    df = df.loc[df["slide_submitter_id"].isin(available_slides)]
    # Cleaning
    columns_needed = [
        "case_id",
        "age_at_index",
        "project_id",
        "days_to_death",
        "vital_status",
        "days_to_last_follow_up",
        "ajcc_pathologic_m",
        "ajcc_pathologic_n",
        "ajcc_pathologic_stage",
        "ajcc_pathologic_t",
        "primary_diagnosis",
        "year_of_diagnosis",
        "slide_submitter_id",
        "case_submitter_id",
        "treatment_type",
    ]
    df_filter = df[columns_needed]
    df_filter = df_filter.drop_duplicates(keep="first")
    df_filter.reset_index(drop=True, inplace=True)
    df_filter["durations"] = df_filter["days_to_last_follow_up"].copy()
    df_filter.loc[df_filter["vital_status"] == "Dead", "durations"] = df_filter.loc[
        df_filter["vital_status"] == "Dead", "days_to_death"
    ].values
    # Censor the index where days_to_death is na but the patient died, replace the na value with days_to_last_follow_up
    index = df_filter[df_filter["durations"].isna()].index
    df_filter.loc[df_filter.index.isin(list(index)), "durations"] = df_filter.loc[
        df_filter.index.isin(index), "days_to_last_follow_up"
    ].values
    df_filter["durations"] = df_filter["durations"].astype("float")
    # replace negative durations with absolute values
    df_filter.loc[df_filter["durations"] < 0, "durations"] = np.abs(
        df_filter.loc[df_filter["durations"] < 0, "durations"].values
    )
    df_filter["vital_status"] = (df_filter["vital_status"] == "Dead") * 1
    df_filter["durations"] = df_filter["durations"] / 30.44
    df_filter.drop(columns=["days_to_death", "days_to_last_follow_up"], inplace=True)
    # drop na for columns in labelset
    df_filter.dropna(subset=labelset, inplace=True)

    # Get treatment related information
    all_case_ids = df_filter["case_id"].unique()
    for case_id in all_case_ids:
        treatments = df_filter.loc[
            df_filter["case_id"] == case_id, "treatment_type"
        ].values
        treatments = [treatment for treatment in treatments if treatment != "'--"]
        treatments.sort()
        if len(treatments) == 0:
            df_filter.loc[df_filter["case_id"] == case_id, "treatment"] = np.NaN
        else:
            df_filter.loc[df_filter["case_id"] == case_id, "treatment"] = " ".join(
                treatments
            )
    df_filter["pharm_treatment"] = df_filter["treatment"].str.contains(
        "Pharmaceutical", na=False
    )
    df_filter["rad_treatment"] = df_filter["treatment"].str.contains(
        "Radiation", na=False
    )
    df_filter.drop(columns=["treatment_type"], inplace=True)
    df_filter.drop_duplicates(
        subset=["slide_submitter_id", "treatment"], inplace=True, ignore_index=True
    )

    print("Before Processing")
    print(df["primary_diagnosis"].value_counts())
    df_filter = cancer_specific_filter(df_filter, onco_code)
    df_filter.reset_index(drop=True, inplace=True)
    return df_filter

def make_dataset(onco_code, label_path, data_path, genomic_path, labelset, output_path):
    """
    Main function to create the dataset of slide features and clinical labels.
    Splits the data into train, validation, and test sets ensuring stratification
    based on primary diagnosis classes.
    Also ensure only data with genomic information are included in val and test sets.
    """
    Path(output_path).mkdir(exist_ok=True, parents=True)

    data_path_list = glob.glob(f"{data_path}/*.pt")
    df = load_labelset(onco_code, data_path_list, label_path, labelset)
    print(df.head())

    # loading genomic information
    df_gene = pd.read_csv(genomic_path)
    # Take into account slides with genomic information available
    df["gene_availability"] = 0
    df.loc[
        df["case_submitter_id"].isin(df_gene["case_id"].to_list()), "gene_availability"
    ] = 1

    print("After Processing")
    print(df["primary_diagnosis"].value_counts())

    datalist = []

    for i, r in df.iterrows():
        d = r.to_dict()
        slidepath = f"{str(Path(data_path))}/{d['slide_submitter_id']}_featvec.pt"
        d["features_path"] = slidepath
        datalist.append(d)
    idxs = np.arange(len(datalist))
    # group idxs on patient level
    colset = labelset + ["case_id", "gene_availability", "primary_class"]
    case_df = df[colset].drop_duplicates().reset_index(drop=True)
    assert len(case_df) == len(case_df["case_id"].unique())

    # Remove cases from stratification without genomic information and ones having primary_class==-1
    # it will be later on added to train set
    # Right now there is not a way to use cases without gene information, but keeping for later use, with llm we can use primary_class==-1
    case_relevant_df = case_df.loc[
        (case_df["gene_availability"] == 1) & (case_df["primary_class"] >= 0)
    ].reset_index(drop=True)
    case_irrelevant_df = case_df.loc[
        (case_df["gene_availability"] == 0) | (case_df["primary_class"] < 0)
    ].reset_index(drop=True)

    train_idxs, test_idxs = train_test_split(
        case_relevant_df["case_id"].tolist(),
        test_size=0.2,
        random_state=0,
        stratify=case_relevant_df["primary_class"].values,
    )
    train_df = case_relevant_df.loc[case_relevant_df["case_id"].isin(train_idxs), :]
    train_idxs, val_idxs = train_test_split(
        train_df["case_id"].tolist(),
        test_size=0.15,
        random_state=0,
        stratify=train_df["primary_class"].values,
    )

    # Add cases from irrelevant_df
    irrelvant_case = case_irrelevant_df["case_id"].tolist()
    print(
        "Case distribution: {}, {}, {}".format(
            len(train_idxs + irrelvant_case), len(val_idxs), len(test_idxs)
        )
    )
    # convert caseids to indices
    train_idxs = list(df.loc[df["case_id"].isin(train_idxs + irrelvant_case)].index)
    val_idxs = list(df.loc[df["case_id"].isin(val_idxs)].index)
    test_idxs = list(df.loc[df["case_id"].isin(test_idxs)].index)

    print(
        "Slide distribution: {}, {}, {}".format(
            len(train_idxs), len(val_idxs), len(test_idxs)
        )
    )
    train_set = [datalist[i] for i in train_idxs]
    val_set = [datalist[i] for i in val_idxs]
    test_set = [datalist[i] for i in test_idxs]

    # Print distribution information
    df_train = df.iloc[train_idxs, :]
    df_val = df.iloc[val_idxs, :]
    df_test = df.iloc[test_idxs, :]

    print("Train statistics")
    print(
        df_train[["case_id", "primary_class", "durations", "vital_status"]]
        .drop_duplicates()
        .describe()
    )
    print(
        df_train[["case_id", "primary_class", "durations", "vital_status"]]
        .drop_duplicates()["vital_status"]
        .value_counts()
    )
    print(
        df_train[["case_id", "primary_class", "durations", "vital_status"]]
        .drop_duplicates()["primary_class"]
        .value_counts()
    )
    print("Val statistics")
    print(
        df_val[["case_id", "primary_class", "durations", "vital_status"]]
        .drop_duplicates()
        .describe()
    )
    print(
        df_val[["case_id", "primary_class", "durations", "vital_status"]]
        .drop_duplicates()["vital_status"]
        .value_counts()
    )
    print(
        df_val[["case_id", "primary_class", "durations", "vital_status"]]
        .drop_duplicates()["primary_class"]
        .value_counts()
    )
    print("Test statistics")
    print(
        df_test[["case_id", "primary_class", "durations", "vital_status"]]
        .drop_duplicates()
        .describe()
    )
    print(
        df_test[["case_id", "primary_class", "durations", "vital_status"]]
        .drop_duplicates()["vital_status"]
        .value_counts()
    )
    print(
        df_test[["case_id", "primary_class", "durations", "vital_status"]]
        .drop_duplicates()["primary_class"]
        .value_counts()
    )

    with open(f"{output_path}/train_{onco_code}_cls_feat.json", "w") as fp:
        json.dump({"data": train_set}, fp)

    with open(f"{output_path}/val_{onco_code}_cls_feat.json", "w") as fp:
        json.dump({"data": val_set}, fp)

    with open(f"{output_path}/test_{onco_code}_cls_feat.json", "w") as fp:
        json.dump({"data": test_set}, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onco_code", default="brca", type=str, help="data code")
    parser.add_argument(
        "--img_input_dir",
        default="./raw_data",
        type=str,
        help="directory containing of raw input files",
    )
    parser.add_argument(
        "--gene_input_dir",
        default="./data",
        type=str,
        help="directory containing processed transcriptomics files",
    )
    parser.add_argument(
        "--feat_input_dir",
        default="./data",
        type=str,
        help="directory containing extracted patch feature vectors from slides in input_dir",
    )

    args = parser.parse_args()
    onco_code = args.onco_code

    ROOT_DIR = Path(__file__).resolve().parent.parent

    make_dataset(
        onco_code=onco_code,
        label_path=Path(args.img_input_dir) / f"TCGA-{onco_code.upper()}",
        data_path=Path(args.feat_input_dir)
        / f"TCGA_MIL_Patches_GigaPath_1MPP_{onco_code.upper()}/",
        genomic_path=Path(args.gene_input_dir)
        / f"tcga_{onco_code}_xena_clean_pathway.csv",
        labelset=["primary_diagnosis", "durations", "vital_status"],
        output_path=ROOT_DIR / f"dataset/json_splits/tcga_{onco_code}",
    )
