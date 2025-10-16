"""
Main training and testing script for ModalTune PanCancer.

Performs multi-modal fusion (Images + Gene + Clinical data) in the aggregator step
using multiple text embeddings as label vectors for cancer classification tasks.

Key Features:
- Multi-task learning across different cancer types using knowledge distillation from text embeddings
- Genomic data incorporation through pathway-based grouping
- Integration of multi-modal features using ModalAdapters
- Training and testing across multiple cancer types
"""

# Standard library imports
from contextlib import nullcontext
from functools import partial
from pathlib import Path

# Third-party imports
import lifelines
import numpy as np
import pandas as pd
import torch
import wandb
from torch import nn
from sklearn import metrics
from torch.distributed.algorithms.join import Join
from contextlib import nullcontext
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

# Local imports
from utils.defaut_args import parser
from utils.constants import NUM_SITES, SITE_LABEL
from utils.test_utils_pancancer import perform_testing_pancancer
from data_utils.datasets import FeaturesGeneTextDataset
from .train_modaltune import MILTextGeneTrainer_multitask, run_trainer

CFD_DIR = Path(__file__).resolve().parent / "model_configs"

class MILTextGeneTrainer_multitask_PC(MILTextGeneTrainer_multitask):
    """WSI-level multiple-instance learning (MIL) trainer with text labels for pancancer"""

    WARMUP_EP = 10
    WARMUP_FACTOR = 20
    # TCGA dataset mapping
    NUM_DATASETS = NUM_SITES # Number of cancer types
    DATASET_MAP = SITE_LABEL # Mapping of cancer types to indices

    def train_one_epoch(self, dataloader):
        """
        Defines one epoch of weakly-supervised MIL training loop
        :param dataloader: torch train dataloader object to iterate
        :return: a tuple of model outputs to compute metrics on
        """
        self.model.train()
        total_loss = 0
        n_iters = 0

        # zero out grads
        self.optimizer.zero_grad()

        # start_time = time.time()
        for images, coords, text, clinical, gene_data, label, case_id in dataloader:
            if clinical is not None:
                clinical = clinical.to(self.device)
            images, text, coords = (
                images.to(self.device),
                text.to(self.device),
                coords.to(self.device),
            )
            if isinstance(gene_data, dict):
                gene_data = {
                    key: value.to(self.device) for key, value in gene_data.items()
                }
            else:
                gene_data = gene_data.to(self.device)
            text = text.squeeze(0)
            text = self.projector(text)
            text = text / text.norm(dim=-1, keepdim=True)

            with Join([self.model]) if self.args.world_size > 1 else nullcontext():
                # prediction is logits for each class for 1 WSI, pred_prob is probability for 1 WSI
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    logit = self.multitask_forward(
                        x=images,
                        coords=coords,
                        genes=gene_data,
                        clinical=clinical,
                        task_ids=[0, 1, 2],
                    )  # logit: (Bxnum_tasks x 256)
                    logit = logit / logit.norm(dim=-1, keepdim=True)
                    # Taken some inspiration from PromptKD repository
                    # 1. KL divergence loss
                    # 2. Normalize feature vectors
                    # 3. Projection layer for text embeddings
                    loss = (
                        self.loss_fn(
                            nn.functional.log_softmax(logit / self.temperature, dim=1),
                            nn.functional.softmax(
                                text[[0, 1, 3], :] / self.temperature, dim=1
                            ),
                        )
                        * (self.temperature**2)
                        * 10
                    )

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            total_loss += loss.item()
            n_iters += 1

        self.scheduler.step()
        # print mean and std of model params lol

        if self.current_epoch % self.args.eval_interval == 0:
            y_true, y_pred, y_probs, c_index, y_sites, y_pred_sites, y_pred_probs = (
                self.LogisticRegression_train(dataloader)
            )
            print(f"Training loss: {total_loss / n_iters}")
            return (
                y_true,
                y_pred,
                y_probs,
                total_loss / n_iters,
                c_index,
                y_sites,
                y_pred_sites,
                y_pred_probs,
            )
        else:
            return None, None, None, total_loss / n_iters, None, None, None, None

    def LogisticRegression_train(self, dataloader):
        """
        Once embeddings are obtained, use logistic regression for evaluation
        """
        self.model.eval()
        # Get embeddings
        x_all = [[] for i in range(self.NUM_DATASETS)]
        y_true_all = [[] for i in range(self.NUM_DATASETS)]
        survival_data_all = [[] for i in range(self.NUM_DATASETS)]
        cancer_site_all = [[] for i in range(self.NUM_DATASETS)]
        df = dataloader.iterable.dataset.df
        with torch.no_grad():
            for image, coords, text, clinical, gene_data, label, case_id in dataloader:
                metadata = df[df["case_id"] == case_id[0]]
                project_id = metadata.iloc[0]["project_id"]
                logreg_idx = self.DATASET_MAP[project_id]
                if clinical is not None:
                    clinical = clinical.to(self.device)
                image, coords = image.to(self.device), coords.to(self.device)
                if isinstance(gene_data, dict):
                    gene_data = {
                        key: value.to(self.device) for key, value in gene_data.items()
                    }
                else:
                    gene_data = gene_data.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    logit = self.multitask_forward(
                        x=image, coords=coords, genes=gene_data, task_ids=[0]
                    )
                if isinstance(label, torch.Tensor):
                    label = label.tolist()
                y_true_all[logreg_idx].append(str(label[0]))
                survival_data_all[logreg_idx].append(
                    (metadata.iloc[0][["vital_status", "durations"]].to_dict())
                )
                x_all[logreg_idx].append(logit.ravel().cpu().numpy())
                cancer_site_all[logreg_idx].append(logreg_idx)

        lr_all = []
        cph_all = []
        pred_train_all = []
        pred_probs_all = []
        c_index_all = []
        y_true_out_all = []

        # Train task specific linear fitters seperately for each cancer site
        for x, y_true, survival_data in zip(x_all, y_true_all, survival_data_all):
            # Diagnosis
            y_true = np.asarray(y_true, int).ravel()
            x = np.array(x)
            x_filter, y_true = self.filter_labelset(x, y_true)
            # Logistic Regression against standard labels
            lr_eval = LogisticRegression(
                random_state=0, max_iter=200, solver="liblinear"
            )
            lr_eval.fit(x_filter, y_true)
            pred_train = lr_eval.predict(x_filter)
            pred_probs = lr_eval.predict_proba(x_filter)
            # survival
            cph = lifelines.fitters.coxph_fitter.CoxPHFitter(penalizer=0.1)
            df_survival_data = pd.DataFrame(survival_data)
            df_model = pd.DataFrame(x, columns=list(np.arange(x.shape[1]).astype(str)))
            df_model["durations"] = df_survival_data["durations"]
            df_model["vital_status"] = df_survival_data["vital_status"]
            df_model.dropna(inplace=True)
            cph.fit(df_model, duration_col="durations", event_col="vital_status")
            c_index = cph.score(df_model, scoring_method="concordance_index")

            lr_all.append(lr_eval)
            cph_all.append(cph)
            pred_train_all.append(pred_train.tolist())
            pred_probs_all.append(pred_probs.tolist())
            c_index_all.append(c_index)
            y_true_out_all.append(y_true.tolist())

        self.lr_eval = lr_all
        self.cph = cph_all

        x_sites = np.concatenate(x_all)
        y_cancer_sites = np.concatenate(cancer_site_all)
        self.lr_cancersite = LogisticRegression(
            random_state=0, max_iter=200, solver="liblinear"
        )
        self.lr_cancersite.fit(x_sites, y_cancer_sites)
        pred_cancer_sites = self.lr_cancersite.predict(x_sites).tolist()
        pred_cancer_probs = self.lr_cancersite.predict_proba(x_sites).tolist()

        return (
            y_true_out_all,
            pred_train_all,
            pred_probs_all,
            c_index_all,
            y_cancer_sites.tolist(),
            pred_cancer_sites,
            pred_cancer_probs,
        )

    def evaluate(self, dataloader, stage):
        """
        Evaulate weakly-supervised MIL
        :param dataloader: torch dataloader object to iterate through
        :param stage: either 'val' or 'test' based on the eval stage
        :return: a tuple of model outputs to compute metrics on
        """
        self.model.eval()

        with torch.no_grad():

            x_logits_all = [[] for i in range(self.NUM_DATASETS)]
            y_true_all = [[] for i in range(self.NUM_DATASETS)]
            survival_data_all = [[] for i in range(self.NUM_DATASETS)]
            cancer_site_all = [[] for i in range(self.NUM_DATASETS)]
            total_loss = 0
            n_iters = 0
            df = dataloader.iterable.dataset.df

            for image, coords, text, clinical, gene_data, label, case_id in dataloader:
                if clinical is not None:
                    clinical = clinical.to(self.device)
                image, text, coords = (
                    image.to(self.device),
                    text.to(self.device),
                    coords.to(self.device),
                )
                metadata = df[df["case_id"] == case_id[0]]
                project_id = metadata.iloc[0]["project_id"]
                logreg_idx = self.DATASET_MAP[project_id]
                if isinstance(gene_data, dict):
                    gene_data = {
                        key: value.to(self.device) for key, value in gene_data.items()
                    }
                else:
                    gene_data = gene_data.to(self.device)
                text = text.squeeze(0)
                text = self.projector(text)
                text = text / text.norm(dim=-1, keepdim=True)

                # prediction is logits for each class for 1 WSI, pred_prob is probability for 1 WSI
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    logit = self.multitask_forward(
                        x=image,
                        coords=coords,
                        genes=gene_data,
                        clinical=clinical,
                        task_ids=[0],
                    )
                    logit_change = logit / logit.norm(dim=-1, keepdim=True)
                    loss = (
                        self.loss_fn(
                            nn.functional.log_softmax(
                                logit_change / self.temperature, dim=1
                            ),
                            nn.functional.softmax(
                                text[[0], :] / self.temperature, dim=1
                            ),
                        )
                        * (self.temperature**2)
                        * 10
                    )

                x_logits_all[logreg_idx].append(logit.ravel().cpu().numpy())
                if isinstance(label, torch.Tensor):
                    label = label.tolist()
                y_true_all[logreg_idx].append(str(label[0]))
                survival_data_all[logreg_idx].append(
                    (metadata.iloc[0][["vital_status", "durations"]].to_dict())
                )
                cancer_site_all[logreg_idx].append(logreg_idx)
                total_loss += loss.item()
                n_iters += 1

            print(f"Validation loss: {total_loss / n_iters}")
            # NO-op if epochs not divisible by some specified constant
            if (stage == "val") and (self.current_epoch % self.args.eval_interval != 0):
                return None, None, None, total_loss / n_iters, None, None, None

            if stage == "test":
                # Need to get the linear models for the best model
                self.cph, self.lr_eval = None, None
                train_iter = tqdm(self.get_eval_iterator(self.train_data["data"]))
                _, _, _, _, _, _, _ = self.LogisticRegression_train(train_iter)

            y_pred_all = []
            y_probs_all = []
            c_index_all = []
            y_true_out_all = []

            for i, (x_logits, y_true, survival_data) in enumerate(
                zip(x_logits_all, y_true_all, survival_data_all)
            ):
                x_logits = np.array(x_logits)
                x_logits_filter, y_true = self.filter_labelset(
                    x_logits, np.asarray(y_true, int).ravel()
                )

                y_pred = self.lr_eval[i].predict(x_logits_filter)
                y_probs = self.lr_eval[i].predict_proba(x_logits_filter)
                # survival
                df_survival_data = pd.DataFrame(survival_data)
                df_model = pd.DataFrame(
                    x_logits, columns=list(np.arange(x_logits.shape[1]).astype(str))
                )
                df_model["durations"] = df_survival_data["durations"]
                df_model["vital_status"] = df_survival_data["vital_status"]
                df_model.dropna(inplace=True)
                c_index = self.cph[i].score(
                    df_model, scoring_method="concordance_index"
                )

                y_pred_all.append(y_pred.tolist())
                y_probs_all.append(y_probs.tolist())
                c_index_all.append(c_index)
                y_true_out_all.append(y_true.tolist())

            x_sites = np.concatenate(x_logits_all)
            y_cancer_sites = np.concatenate(cancer_site_all)
            y_pred_cancer_sites = self.lr_cancersite.predict(x_sites).tolist()
            y_probs_cancer_sites = self.lr_cancersite.predict_proba(x_sites).tolist()

            return (
                y_true_out_all,
                y_pred_all,
                y_probs_all,
                total_loss / n_iters,
                c_index_all,
                y_cancer_sites,
                y_pred_cancer_sites,
                y_probs_cancer_sites,
            )

    def compute_metrics(self, outputs, stage):
        """
        Compute relevant metrics for slide-based weakly supervised learning to track progress
        :param outputs: tuple of model outputs to compute metrics on
        :param stage: stage of experiment denoted by a string
        :return: a dictionary of metrics to be tracked and a key metric used to choose best validation model
        """
        metrics_dict = {}
        (
            y_true_wsi_all,
            y_pred_wsi_all,
            y_probs_wsi_all,
            cls_loss,
            c_index_all,
            y_true_cancer_site,
            y_pred_cancer_site,
            y_probs_cancer_site,
        ) = outputs

        # NO-OP if inputs are none
        if y_true_wsi_all is None:
            return metrics_dict, -1.0

        acc_wsi_all = []
        bal_acc_wsi_all = []
        recall_wsi_all = []
        precision_wsi_all = []
        f1_wsi_all = []
        for i in range(self.NUM_DATASETS):
            if self.args.num_classes[i] < 2:
                average = "binary"
            else:
                average = None

            # get general confusion matrix metrics
            acc_wsi = metrics.accuracy_score(y_true_wsi_all[i], y_pred_wsi_all[i])
            bal_acc_wsi = metrics.balanced_accuracy_score(
                y_true_wsi_all[i], y_pred_wsi_all[i]
            )
            recall_wsi = metrics.recall_score(
                y_true_wsi_all[i], y_pred_wsi_all[i], average=average
            )
            precision_wsi = metrics.precision_score(
                y_true_wsi_all[i], y_pred_wsi_all[i], average=average
            )
            f1_wsi = metrics.f1_score(
                y_true_wsi_all[i], y_pred_wsi_all[i], average=average
            )

            acc_wsi_all.append(acc_wsi)
            bal_acc_wsi_all.append(bal_acc_wsi)
            recall_wsi_all.append(recall_wsi)
            precision_wsi_all.append(precision_wsi)
            f1_wsi_all.append(f1_wsi)

        acc_cancer_site = metrics.accuracy_score(y_true_cancer_site, y_pred_cancer_site)
        bal_acc_cancer_site = metrics.balanced_accuracy_score(
            y_true_cancer_site, y_pred_cancer_site
        )

        # return metrics dict for wandb with confusion matrix + ROC
        metrics_dict.update(
            {
                f"{stage}_cls_acc": np.mean(acc_wsi_all),
                f"{stage}_bal_cls_acc": np.mean(bal_acc_wsi_all),
                f"{stage}_c_index": np.mean(c_index_all),
                f"{stage}_cancer_site_acc": acc_cancer_site,
                f"{stage}_cancer_site_bal_acc": bal_acc_cancer_site,
                f"{stage}_cls_acc_all": acc_wsi_all,
                f"{stage}_bal_cls_acc_all": bal_acc_wsi_all,
                f"{stage}_c_index_all": c_index_all,
                f"{stage}_cls_loss": cls_loss,
                f"{stage}_cls_recall_all": recall_wsi_all,
                f"{stage}_cls_precision_all": precision_wsi_all,
                f"{stage}_cls_f1_all": f1_wsi_all,
            }
        )
        # accuracy as key metric
        key_metric = metrics_dict[f"{stage}_bal_cls_acc"]
        return metrics_dict, key_metric

    def deploy_mil(self):
        """
        Deploys pretrained MIL model on a new test set
        1. Loads the best model weights
        2. Generates feature embeddings for each WSI
        3. Performs logistic regression for classification tasks and cox regression for survival tasks
        Additionally saves the feature embeddings and labels for future use
        """
        assert (
            self.args.world_size <= 1
        ), "Distributed training not supported with this eval function!"

        self.init_model_and_optimizer()
        self.Dataset_Class = partial(
            FeaturesGeneTextDataset,
            case_wise=True,
            return_case=True,
            gene_group_defination=self.gene_group_defination,
            threshold=self.THRESHOLD,
        )

        wandb.init(mode="disabled")

        # load dataset
        train_iter = self.get_eval_iterator(self.train_data["data"])
        val_iter = self.get_eval_iterator(self.val_data["data"])
        test_iter = self.get_eval_iterator(self.test_data["data"])

        # load model
        state_dict = torch.load(self.args.eval_weights, map_location="cpu")
        miss, unexpected = self.model.load_state_dict(state_dict)
        print(
            f"Testing with model saved weights: {self.args.eval_weights}\nMissed: {miss}\nUnexpected: {unexpected}"
        )
        self.model.eval()

        label_set = [
            "vital_status",
            "durations",
            "primary_class",
            "primary_diagnosis",
            "ajcc_pathologic_stage",
            "ajcc_pathologic_t",
            "ajcc_pathologic_n",
            "ajcc_pathologic_m",
            "project_id",
        ]
        x_train, df_train, x_val, df_val, x_test, df_test = self.get_features(
            train_iter, val_iter, test_iter, label_set
        )
        print("Feature vectors saved...")
        perform_testing_pancancer(x_train, df_train, x_test, df_test, penalizer=0.1)


if __name__ == "__main__":
    # add new args here
    parser.add_argument(
        "--num_classes",
        default="2,2,2,3",
        type=str,
        help="number of classification classes for evaluation",
    )
    parser.add_argument(
        "--model_config",
        default="conch_ViT-B-16_prompt",
        type=str,
        help="model config for pretrained ViT",
    )
    parser.add_argument(
        "--mil_name",
        default="abmil",
        type=str,
        help="choose between different aggregators",
    )
    parser.add_argument(
        "--text_location",
        default="./data/BRCA_textembeddings_conch_ViT-B-16_all.pt",
        type=str,
        help="location of text embeddings",
    )
    parser.add_argument(
        "--gc", default=1, type=int, help="number of gradient accumulation steps"
    )
    parser.add_argument(
        "--threshold",
        default=25000,
        type=int,
        help="maximum number of patches to consider",
    )
    parser.add_argument(
        "--num_tasks",
        default=4,
        type=int,
        help="Number of tasks to train the model on simultaneously",
    )
    parser.add_argument(
        "--genomics_csv_path",
        type=str,
        default="./data/genomics.csv",
        help="Location for mRNA sequencing data in csv format",
    )
    parser.add_argument(
        "--clinical_location",
        default="",
        type=str,
        help="location of simple clinical features",
    )

    # args for evaluating on new data
    parser.add_argument(
        "--eval_only",
        default=0,
        type=int,
        help="set to 1 if looking to only eval on a new dataset with pretrained weights",
    )
    parser.add_argument(
        "--eval_weights", type=str, help="path to ensemble model weights for only eval"
    )
    parser.add_argument(
        "--eval_name",
        type=str,
        default="mil",
        help="name for saving feature embeddings for evaluation",
    )

    args = parser.parse_args()
    args.num_classes = [int(x) for x in args.num_classes.split(",")]

    # Run experiments
    modaltune_trainer = MILTextGeneTrainer_multitask_PC
    run_trainer(args, modaltune_trainer)
