"""
Script for testing ModalTune PanCancer generated feature vectors across different tasks and task prompts
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

# survival analysis
import lifelines

# Linear model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from .constants import TASK_IDS, NUM_SITES, SITE_LABEL, PROJECT_ID_MAP


def filter_labelset(x, y):
    """
    Only select indicies which has certain labels available
    By default while making dataset, labels with y=-1 consists of rare labels not included for classification
    """
    idx = np.where(y >= 0)[0]
    x_new = x[idx, :]
    y_new = y[idx]
    return x_new, y_new

def test_label(x_train, y_train, x_test, y_test):
    """
    Test the label prediction(classification) performance of the model
    """
    y_train_diag = np.asarray(y_train, int).ravel()
    y_test_diag = np.asarray(y_test, int).ravel()
    x_test_diag = np.array(x_test)
    x_train = np.array(x_train)
    x_train_diag, y_train_diag = filter_labelset(x_train, y_train_diag)
    clf = LogisticRegression(random_state=0, max_iter=200, solver="liblinear").fit(
        x_train_diag, y_train_diag
    )
    y_pred_test = clf.predict(x_test_diag)
    y_pred_train = clf.predict(x_train_diag)
    print(np.unique(y_train_diag, return_counts=True))
    print(np.unique(y_test_diag, return_counts=True))
    print(
        "Train accuracy: {}\nTest accuracy: {}".format(
            accuracy_score(y_train_diag, y_pred_train),
            accuracy_score(y_test_diag, y_pred_test),
        )
    )
    print(
        "Train balanced accuracy: {}\nTest balanced accuracy: {}".format(
            balanced_accuracy_score(y_train_diag, y_pred_train),
            balanced_accuracy_score(y_test_diag, y_pred_test),
        )
    )
    print(
        "Train confusion matrix: {}".format(
            confusion_matrix(y_train_diag, y_pred_train)
        )
    )
    print(
        "Test confusion matrix: {}".format(confusion_matrix(y_test_diag, y_pred_test))
    )
    return clf

def get_survival_model(x_train, df_train, penalizer=0.1, cluster_site=False):  
    # survival
    cph = lifelines.fitters.coxph_fitter.CoxPHFitter(penalizer=penalizer)

    df_model = pd.DataFrame(
        x_train, columns=list(np.arange(x_train.shape[1]).astype(str))
    )
    df_model["durations"] = df_train["durations"]
    df_model["vital_status"] = df_train["vital_status"]
    if cluster_site:
        df_model["project_id"] = df_train["project_id"]
        cluster_col = "project_id"
    else:
        cluster_col = None
    df_model.dropna(inplace=True)
    cph.fit(
        df_model, duration_col="durations", event_col="vital_status", strata=cluster_col
    )
    return cph

def get_test_score(cph_model, x_test, df_test, cluster_site=False):
    df_model_test = pd.DataFrame(
        x_test, columns=list(np.arange(x_test.shape[1]).astype(str))
    )
    df_model_test["vital_status"] = df_test["vital_status"]
    df_model_test["durations"] = df_test["durations"]
    if cluster_site:
        df_model_test["project_id"] = df_test["project_id"]
    df_model_test.dropna(inplace=True)
    print(
        "Test score: {}".format(
            cph_model.score(df_model_test, scoring_method="concordance_index")
        )
    )

def test_survival(x_train, df_train, x_test, df_test, penalizer=0.1, cluster_site=False):
    """
    Performs survival analysis using Cox Proportional Hazards model from lifelines package
    Args:
        x_train: np.array
            training feature vectors [N, dim]
        df_train: pd.DataFrame
            training clinical dataframe with 'durations' and 'vital_status' columns
        x_test: np.array
            testing feature vectors [M, dim]
        df_test: pd.DataFrame
            testing clinical dataframe with 'durations' and 'vital_status' columns
        penalizer: float
            penalizer for CoxPHFitter
        cluster_site: bool
            whether to cluster by cancer site while training the survival model
    """  
    cph = get_survival_model(x_train, df_train, penalizer, cluster_site)
    get_test_score(cph, x_test, df_test, cluster_site)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_name", type=str, required=True, help="name of the stored embeddings"
    )
    args = parser.parse_args()
    return args


def perform_testing_pancancer(x_train, df_train, x_test, df_test, penalizer=0.1):
    """
    Performs testing for pancancer dataset. Additionaly performs testing for individual cancer sites
    Args:
        x_train: np.array
            training feature vectors [N, n_tasks, dim]
        df_train: pd.DataFrame
            training clinical dataframe with 'durations' and 'vital_status' columns
            and 'primary_class' column for diagnosis classification
            and 'project_id' column for cancer site classification
        x_test: np.array
            testing feature vectors [M, n_tasks, dim]
        df_test: pd.DataFrame
            testing clinical dataframe with 'durations' and 'vital_status' columns
            and 'primary_class' column for diagnosis classification
            and 'project_id' column for cancer site classification
        penalizer: float
            penalizer for CoxPHFitter
    """
    embedding_text = list(TASK_IDS.values())
    print(
        f"Performing testing on multiple tasks based on embeddings: {embedding_text}..."
    )
    print("Training data shape: {}".format(x_train.shape))
    print("Testing data shape: {}".format(x_test.shape))

    # get pooled survival model merging all sites
    pooled_cph = []
    n_tasks = x_train.shape[1]
    for i in range(n_tasks):
        pooled_cph.append(
            get_survival_model(
                x_train[:, i, :], df_train, penalizer=penalizer, cluster_site=False
            )
        )

    for key in PROJECT_ID_MAP.keys():

        print(f"======================== {key} ========================")
        project_ids = PROJECT_ID_MAP[key]
        df_train_proj = df_train[df_train["project_id"].isin(project_ids)].reset_index(
            drop=True
        )
        df_test_proj = df_test[df_test["project_id"].isin(project_ids)].reset_index(
            drop=True
        )
        x_train_proj = x_train[df_train["project_id"].isin(project_ids)]
        x_test_proj = x_test[df_test["project_id"].isin(project_ids)]

        if len(df_train_proj) == 0:
            continue
        print(len(df_train_proj), len(df_test_proj))
        print(x_train_proj.shape, x_test_proj.shape)

        print("######## Survival Prediction ########")
        n_tasks = x_train_proj.shape[1]
        for i in range(n_tasks):
            print(f"Model - {embedding_text[i]}")
            test_survival(
                x_train_proj[:, i, :],
                df_train_proj,
                x_test_proj[:, i, :],
                df_test_proj,
                penalizer=0.1,
            )

        print("######## Pooled Survival Prediction ########")
        for i in range(n_tasks):
            print(f"Model - {embedding_text[i]}")
            get_test_score(
                pooled_cph[i], x_test_proj[:, i, :], df_test_proj, cluster_site=False
            )

        print("######## Diagnosis ########")
        if key == "TCGA-RCC":
            labelset = ["0", "1", "2"]
        else:
            labelset = ["0", "1"]
        for i in range(n_tasks):
            print(f"Model - {embedding_text[i]}")
            test_label(
                x_train_proj[:, i, :],
                df_train_proj["primary_class"].values,
                x_test_proj[:, i, :],
                df_test_proj["primary_class"].values,
                labelset,
            )

    print(
        "===================================== Cancer site prediction ====================================="
    )
    n_tasks = x_train.shape[1]
    train_sites = df_train["project_id"].map(SITE_LABEL).values
    test_sites = df_test["project_id"].map(SITE_LABEL).values
    for i in range(n_tasks):
        print(f"Model - {embedding_text[i]}")
        test_label(
            x_train[:, i, :],
            train_sites,
            x_test[:, i, :],
            test_sites,
            np.asarray(np.arange(NUM_SITES), "str"),
        )


if __name__ == "__main__":
    ### BEFORE RUNNING DIRECTLY ENSURE FEATURE EMBEDDINGS ARE AVAILABLE FROM MAIN SCRIPT

    results_dir = Path(__file__).resolve().parent / "results/data"
    args = get_args()
    identifier = "_" + args.eval_name

    print("Loading from : {}".format(identifier))
    df_train = pd.read_csv(results_dir / ("train" + identifier + ".csv"))
    df_test = pd.read_csv(results_dir / ("test" + identifier + ".csv"))
    with open(str(results_dir / ("x_feats" + identifier + ".npy")), "rb") as file:
        x_train = np.load(file)
        x_val = np.load(file)
        x_test = np.load(file)

    perform_testing_pancancer(x_train, df_train, x_test, df_test, penalizer=0.1)
