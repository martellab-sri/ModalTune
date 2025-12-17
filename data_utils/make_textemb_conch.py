"""
This script is about using pretrained LLMs to construct text embeddings from tabular data.
The loads data from pre-made datasets from make_dataset.py
"""

import sys
import argparse
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
import numpy as np
import pandas as pd
import torch

from utils.constants import CONCH_CFG, CONCH_CHECKPOINT_PATH

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cancer_code = {
    "BRCA": "breast",
    "BLCA": "bladder urothelial",
    "COADREAD": "colorectal",
    "GBMLGG": "brain",
    "NSCLC": "lung",
    "RCC": "kidney",
    "STAD": "stomach",
    "UCEC": "uterus",
}

stage_mapper = {
    "Stage I": "stage one",
    "Stage II": "stage two",
    "Stage III": "stage three",
    "Stage IV": "stage four",
    "Stage X": "stage cannot be determined",
}

tumor_stage_mapper = {
    "T0": "no tumor detected",
    "T1": "tumor stage one",
    "T2": "tumor stage two",
    "T3": "tumor stage three",
    "T4": "tumor stage four",
    "TX": "tumor stage cannot be assessed",
}

node_stage_mapper = {
    "N0": "cancer has not spread to lymph nodes",
    "N1": "node stage one",
    "N2": "node stage two",
    "N3": "node stage three",
    "NX": "node spread cannot be assessed",
}
metastasis_stage_mapper = {
    "M0": "no metastasis detected",
    "M1": "cancer has spread to distant organs",
    "MX": "metastasis status cannot be assessed",
}


def get_intervals(df, n_bins=4):
    """
    converts duration into bucket of n_bins
    """
    patients_df = df.drop_duplicates(["case_id"]).copy()
    disc_labels, q_bins = pd.qcut(
        patients_df["durations"], q=n_bins, retbins=True, labels=False
    )
    q_bins[-1] = patients_df["durations"].max() + 1e-6
    q_bins[0] = patients_df["durations"].min() - 1e-6
    return torch.tensor(q_bins)


def preprocess_data(df_filter, qbins):
    """
    Preprocess the clinical data by cleaning and mapping stages and bucketing durations.
    """
    assert len(df_filter) == len(df_filter["case_id"].unique())
    # Remove NOS, and convert to small
    df_analyze = df_filter.copy()

    df_analyze.loc[
        ~df_analyze["ajcc_pathologic_stage"].isna(), "ajcc_pathologic_stage"
    ] = df_filter.loc[
        ~df_filter["ajcc_pathologic_stage"].isna(), "ajcc_pathologic_stage"
    ].apply(
        lambda x: stage_mapper[
            x.replace("A", "")
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
        ]
    )
    df_analyze.loc[
        ~df_analyze["ajcc_pathologic_t"].isna(), "ajcc_pathologic_t"
    ] = df_filter.loc[
        ~df_filter["ajcc_pathologic_t"].isna(), "ajcc_pathologic_t"
    ].apply(
        lambda x: tumor_stage_mapper[
            x.replace("A", "")
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
        ]
    )
    df_analyze.loc[
        ~df_analyze["ajcc_pathologic_n"].isna(), "ajcc_pathologic_n"
    ] = df_filter.loc[
        ~df_filter["ajcc_pathologic_n"].isna(), "ajcc_pathologic_n"
    ].apply(
        lambda x: node_stage_mapper[
            x.replace("A", "")
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
        ]
    )
    df_analyze.loc[
        ~df_analyze["ajcc_pathologic_m"].isna(), "ajcc_pathologic_m"
    ] = df_filter.loc[
        ~df_filter["ajcc_pathologic_m"].isna(), "ajcc_pathologic_m"
    ].apply(
        lambda x: metastasis_stage_mapper[
            x.replace("A", "")
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
        ]
    )
    df_analyze["label"] = (
        (
            torch.bucketize(
                torch.tensor(df_analyze["durations"].values), qbins[:-1], right=True
            )
            - 1
        )
        .numpy()
        .astype(int)
    )
    df_analyze.reset_index(drop=True, inplace=True)
    return df_analyze


def embedder_process(model, text_prompts):
    """
    Process text prompts to generate embeddings using the provided model."""
    tokenizer = get_tokenizer()
    tokenized_prompts = tokenize(texts=text_prompts, tokenizer=tokenizer).to(DEVICE)
    with torch.no_grad():
        text_embedings = model.encode_text(tokenized_prompts)
    return text_embedings.cpu()


def generate_prompts(df_analyze, model, sent_label, onco_code):
    """
    Generate text prompts based on clinical data and obtain their embeddings.
    """
    text_prompts = []
    diagnosis_prompts = []
    stage_prompts = []
    survival_prompts = []
    event = {0: "was censored", 1: "died"}

    for i in range(len(df_analyze)):
        diag_sent = ""
        stage_sent = ""
        t_sent = ""
        n_sent = ""
        m_sent = ""
        surv_sent = ""
        onco_sent = f"Cancer location: {cancer_code[onco_code]};"
        if not pd.isna(df_analyze.loc[i, "primary_diagnosis"]):
            # diag_sent = f"an H&E image of {cancer_code[ONCO_CODE]} cancer; Cancer diagnosis: {df_analyze.loc[i,'primary_diagnosis']};"
            diag_sent = f"Cancer diagnosis: {df_analyze.loc[i,'primary_diagnosis']};"
        if not pd.isna(df_analyze.loc[i, "ajcc_pathologic_stage"]):
            stage_sent = f"Overall stage: {df_analyze.loc[i,'ajcc_pathologic_stage']};"
        if not pd.isna(df_analyze.loc[i, "ajcc_pathologic_m"]):
            m_sent = (
                f"Distant metastasis status: {df_analyze.loc[i,'ajcc_pathologic_m']};"
            )
        if not pd.isna(df_analyze.loc[i, "ajcc_pathologic_n"]):
            n_sent = f"Lymph node status: {df_analyze.loc[i,'ajcc_pathologic_n']};"
        if not pd.isna(df_analyze.loc[i, "ajcc_pathologic_t"]):
            t_sent = f"Tumor stage status: {df_analyze.loc[i,'ajcc_pathologic_t']};"
        if not pd.isna(df_analyze.loc[i, "durations"]):
            surv_sent = f"Survival status: The patient {event[df_analyze.loc[i,'vital_status']]} {sent_label[df_analyze.loc[i,'label']]}"
        text_prompts.append(
            f"{onco_sent} {diag_sent} {stage_sent} {t_sent} {n_sent} {m_sent} {surv_sent}"
        )
        diagnosis_prompts.append(f"{onco_sent} {diag_sent}")
        stage_prompts.append(f"{onco_sent} {stage_sent} {t_sent} {n_sent} {m_sent}")
        survival_prompts.append(
            f"{onco_sent} {stage_sent} {t_sent} {n_sent} {m_sent} {surv_sent}"
        )

    text_embeddings1 = embedder_process(model, text_prompts)
    text_embeddings2 = embedder_process(model, diagnosis_prompts)
    text_embeddings3 = embedder_process(model, stage_prompts)
    text_embeddings4 = embedder_process(model, survival_prompts)

    case_ids = df_analyze["case_id"].tolist()
    text_embeddings = torch.stack(
        (text_embeddings1, text_embeddings2, text_embeddings3, text_embeddings4), dim=1
    )
    print("Text examples")
    print(text_prompts[-5:])
    return text_embeddings, case_ids

def make_dataset(model, onco_code, output_file):
    """
    Main function to create the dataset of text embeddings.
    Creates a Nx4x512 vector
    where 4 represents the different tasks
    0: General
    1: Primary diagnosis
    2: Stage
    3: Survival
    """
    # load main dataframe
    with open(
        str(
            ROOT_DIR
            / f"dataset/json_splits/tcga_{onco_code.lower()}/train_{onco_code.lower()}_cls_feat.json"
        ),
        "r",
    ) as file:
        train_datalist = json.load(file)["data"]
        train_df = pd.DataFrame(train_datalist)
    with open(
        str(
            ROOT_DIR
            / f"dataset/json_splits/tcga_{onco_code.lower()}/val_{onco_code.lower()}_cls_feat.json"
        ),
        "r",
    ) as file:
        val_datalist = json.load(file)["data"]
        val_df = pd.DataFrame(val_datalist)
    with open(
        str(
            ROOT_DIR
            / f"dataset/json_splits/tcga_{onco_code.lower()}/test_{onco_code.lower()}_cls_feat.json"
        ),
        "r",
    ) as file:
        test_datalist = json.load(file)["data"]
        test_df = pd.DataFrame(test_datalist)
    df = pd.concat((train_df, val_df, test_df), axis=0)
    qbins_trch = get_intervals(df, n_bins=N_BINS)
    qbins = np.round(qbins_trch.numpy()).astype(int)

    df.drop_duplicates(subset="case_id", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = preprocess_data(df, qbins_trch)
    # bining
    sent_label = {}
    sent_label[0] = f"before {qbins[1]} months"
    sent_label[len(qbins) - 1] = f"after {qbins[len(qbins)-1]} months"
    for i in range(1, len(qbins) - 1):
        sent_label[i] = f"between {qbins[i]} and {qbins[i+1]} months"
    print("Sentence bins: {}".format(sent_label))
    # generate prompts and their text embeddings
    text_embeddings, case_ids = generate_prompts(df, model, sent_label, onco_code)
    # save embeddings per case id
    save_dict = {case_ids[i]: text_embeddings[i, :, :] for i in range(len(case_ids))}
    torch.save(save_dict, output_file)
    print("Done...")


# Binning time intervals into 4 bins
N_BINS = 4
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onco_code", default="brca", type=str, help="project code")
    parser.add_argument(
        "--output_dir",
        default="./data",
        type=str,
        help="directory for storing processed files",
    )
    args = parser.parse_args()
    onco_code = args.onco_code.upper()
    output_dir = Path(args.output_dir)

    # Load LLM model
    model, preprocess = create_model_from_pretrained(
        CONCH_CFG, CONCH_CHECKPOINT_PATH, device=DEVICE
    )
    _ = model.eval()
    # Make a single .pt file
    make_dataset(
        model, onco_code, output_dir / f"{onco_code}_textembeddings_{CONCH_CFG}.pt"
    )  # different binning style
