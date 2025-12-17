"""
Main training and testing script for ModalTune.

Performs multi-modal fusion (Images + Gene + Clinical data) in the aggregator step
using multiple text embeddings as label vectors for cancer classification tasks.

Key Features:
- Multi-task learning across different cancer types using knowledge distillation from text embeddings
- Genomic data incorporation through pathway-based grouping
- Integration of multi-modal features using ModalAdapters
"""

# Standard library imports
import argparse
import json
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path

# Third-party imports
import lifelines
import numpy as np
import pandas as pd
import torch
import wandb
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.distributed.algorithms.join import Join
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler

# Local imports
from data_utils.datasets import FeaturesGeneTextDataset
from models.aggregators import Aggregator
from models.genomic_utils.define_gene_groups import pathway_gene_groups
from utils.base_trainer import Trainer
from utils.defaut_args import parser
from utils.test_utils_modaltune import perform_testing

CFD_DIR = Path(__file__).resolve().parent / "model_configs"

class Projection_layer(nn.Module):
    """
    A simple projection layer to project text embeddings to a different dimension.
    """
    def __init__(self, input_dim=512, out_dim=256):
        super(Projection_layer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, out_dim, 1),
            nn.LayerNorm([out_dim,1,1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 1)
        )
    def forward(self, input_feat):
        final_feat = self.conv1(input_feat.unsqueeze(-1).unsqueeze(-1))
        return final_feat.squeeze(-1).squeeze(-1)

class MILTextGeneTrainer_multitask(Trainer):
    """WSI-level multiple-instance learning (MIL) trainer with text labels"""

    WARMUP_EP = 10
    WARMUP_FACTOR = 20
    #patch threshold

    def __init__(self, args):
        """
        Initialize the MIL trainer
        :param args: argparse arguments entered for experiment
        """
        self.run_task = 'End-to-end tuning WSI-level MIL training'
        print(f'Task={self.run_task}')
        super().__init__(args)

        # 1 WSI at a time for MIL with a limit on number of patches/slide given by threshold
        self.args.batch_size = 1
        self.THRESHOLD = self.args.threshold
        self.args.mode = "feature"

        # Temperature for distillation loss
        self.temperature = 1.0

        #Model configurations
        with open(str(CFD_DIR / self.args.model_config)+".json","r") as file:
            self.model_config = json.load(file)
            print(f"Mode Configuration:\n{self.model_config}")
        
        # loss func for experiments inspired from PromptKD repository (https://github.com/zhengli97/PromptKD)
        self.loss_fn = nn.KLDivLoss(reduction="sum")
        self.loss_fn = self.loss_fn.to(self.device)

        # definintions to create datasets, found in base Trainer class
        gene_df = pd.read_csv(self.args.genomics_csv_path)
        #We group genes as pathways similar to survpath paper, however it can be considered separately
        self.gene_group_defination = pathway_gene_groups(gene_df=gene_df, group_definations=True) 
        
        self.train_transforms = None
        self.eval_transforms = None
        self.Dataset_Class = partial(FeaturesGeneTextDataset,
                                     case_wise=True,
                                     return_case=True,
                                     gene_group_defination=self.gene_group_defination,
                                     threshold=self.THRESHOLD)

        self.scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp, init_scale=2.0**15)
        self.scheduler = None

        #conch text encoder dimension
        input_dim = 512
        
        #We do not finetune the projector, instead use a frozen random projector for dimensionality reduction
        self.projector = Projection_layer(input_dim=input_dim, out_dim=self.model_config["output_dim"]).to(self.device)
        for params in self.projector.parameters():
            params.requires_grad = False

    def init_model_and_optimizer(self):
        """
        Initializes model and optimizer for MIL training
        """

        self.model = Aggregator.create(subclass_name=self.args.mil_name,
                                       gene_group_defination=self.gene_group_defination,
                                       **self.model_config, multi_task=self.args.num_tasks)
        self.model = self.model.to(self.device)

        #Model stats
        total_frozen_param = 0
        total_trainable_param = 0
        for params in self.model.parameters():
            if params.requires_grad:
                total_trainable_param += params.numel()
            else:
                total_frozen_param += params.numel()
        print("Initialized Model...\nTotal number of trainable parameters: {}\nTotal number of frozen parameters: {}".format(total_trainable_param, total_frozen_param))

        # filter for frozen params
        params = [
            {
                "params": filter(lambda p: p.requires_grad, self.model.parameters()),
                "lr": self.args.lr / self.WARMUP_FACTOR
            }
        ]
        self.optimizer = torch.optim.AdamW(
            params,
            weight_decay=self.args.weight_decay,
            betas=(self.args.beta1, self.args.beta2),
        )

        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.args.num_epochs - self.WARMUP_EP)
        self.scheduler = GradualWarmupScheduler(
            self.optimizer, multiplier=self.WARMUP_FACTOR, total_epoch=self.WARMUP_EP, after_scheduler=scheduler_cosine)

    def multitask_forward(self, task_ids:list=None, **kwargs):
        """
        Forward pass for multi-task learning with task-specific embeddings.
        For the set of experiments
            0: General
            1: Diagnosis
            2: Survival prediction
        Args:
            task_ids (list, optional): List of task IDs to process. If None, 
                                    processes all configured tasks.
            **kwargs: Additional arguments passed to the model forward pass
                    including 'images', 'genes', 'clinical_features'
    
        Returns:
            logits [batch_size, num_tasks, output_dim]
        """
        num_tasks = self.args.num_tasks
        logits_all = []
        if self.model.is_multi:
            for tasks in task_ids:
                logits_all.append(self.model(**kwargs,task_token=torch.eye(num_tasks)[tasks].to(self.device)))
            return torch.cat(logits_all,dim=0)
        else:
            return self.model(**kwargs)
    
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
            if n_iters>5:
                break
            if len(clinical):
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
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    #logit is prediction for the text prompt across different task prompts
                    logit = self.multitask_forward(
                        x=images,
                        coords=coords,
                        genes=gene_data,
                        clinical=clinical,
                        task_ids=[0, 1, 2],
                    ) #logit: (Bxnum_tasks x 256)
                    logit = logit / logit.norm(dim=-1, keepdim=True)
                    # Taken some inspiration from PromptKD repository (https://github.com/zhengli97/PromptKD)
                    # 1. KL divergence loss
                    # 2. Normalize feature vectors
                    # 3. Projection layer for text embeddings
                    loss = self.loss_fn(
                        nn.functional.log_softmax(logit / self.temperature, dim=1),
                        nn.functional.softmax(text[[0,1,3],:] / self.temperature, dim=1),
                    ) * (self.temperature ** 2) * 10 

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            total_loss += loss.item()
            n_iters += 1

        self.scheduler.step()
        # print mean and std of model params lol
        
        if self.current_epoch % self.args.eval_interval == 0:
            y_true, y_pred, y_probs, c_index = self.LogisticRegression_train(dataloader)
            print(f"Training loss: {total_loss / n_iters}")
            return y_true, y_pred, y_probs, total_loss / n_iters, c_index
        else:
            return None, None, None, total_loss / n_iters, None

    def get_features(self, train_dataloader, val_dataloader, test_dataloader, label_list):
        """
        Calculates the feature embeddings of tissue section for each case id
        across different task prompts
        """
        self.model.eval()
        #Make a directory inside the model weights folder
        print(f"Saving features at {Path(self.args.output_path) / 'data'}")
        (Path(self.args.output_path) / "data").mkdir(parents=False, exist_ok=True)
        #Get embeddings
        x_train = []
        case_id_train = []
        x_val = []
        case_id_val = []
        x_test = []
        case_id_test = []
        label_list = label_list + ["case_id"]
        with torch.no_grad():
            for images, coords, text, clinical, gene_data, label, case_id in tqdm(train_dataloader):
                if len(clinical):
                    clincial = clincial.to(self.device)
                images, coords = images.to(self.device), coords.to(self.device)
                if isinstance(gene_data,dict):
                    gene_data = {key: value.to(self.device) for key, value in gene_data.items()}
                else:
                    gene_data = gene_data.to(self.device)
                case_id_train.extend(case_id)
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    logit = self.multitask_forward(x=images, coords=coords, genes=gene_data, clinical=clinical, task_ids=[0,1,2])
                x_train.append(logit.cpu().numpy())
            for images, coords, text, clinical, gene_data, label, case_id in tqdm(val_dataloader):
                if len(clinical):
                    clinical = clincial.to(self.device)
                images, coords = images.to(self.device), coords.to(self.device)
                if isinstance(gene_data,dict):
                    gene_data = {key: value.to(self.device) for key, value in gene_data.items()}
                else:
                    gene_data = gene_data.to(self.device)
                case_id_val.extend(case_id)
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    logit = self.multitask_forward(x=images, coords=coords, genes=gene_data, clinical=clinical, task_ids=[0,1,2])
                x_val.append(logit.cpu().numpy())
            for images, coords, text, clinical, gene_data, label, case_id in tqdm(test_dataloader):
                if len(clinical):
                    clinical = clinical.to(self.device)
                images, coords = images.to(self.device), coords.to(self.device)
                if isinstance(gene_data,dict):
                    gene_data = {key: value.to(self.device) for key, value in gene_data.items()}
                else:
                    gene_data = gene_data.to(self.device)
                case_id_test.extend(case_id)
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    logit = self.multitask_forward(x=images, coords=coords, genes=gene_data, clinical=clinical, task_ids=[0,1,2])
                x_test.append(logit.cpu().numpy())
        #Prepare features N x out_dim x 3
        x_train = np.array(x_train)
        x_val = np.array(x_val)
        x_test = np.array(x_test)
        #Prepare labels in form of dataframe
        df_train = train_dataloader.dataset.df
        df_val = val_dataloader.dataset.df
        df_test = test_dataloader.dataset.df
        df_train = df_train.loc[df_train["case_id"].isin(case_id_train),label_list].drop_duplicates(ignore_index=True)
        df_val = df_val.loc[df_val["case_id"].isin(case_id_val),label_list].drop_duplicates(ignore_index=True)
        df_test = df_test.loc[df_test["case_id"].isin(case_id_test),label_list].drop_duplicates(ignore_index=True)

        #Save temp
        print("saving")
        df_train.to_csv(Path(self.args.output_path) / f"data/train_{self.args.eval_name}.csv",index=False)
        df_val.to_csv(Path(self.args.output_path) / f"data/val_{self.args.eval_name}.csv",index=False)
        df_test.to_csv(Path(self.args.output_path) / f"data/test_{self.args.eval_name}.csv",index=False)
        with open(Path(self.args.output_path) / f"data/x_feats_{self.args.eval_name}.npy","wb") as file:
            np.save(file, x_train)
            np.save(file, x_val)
            np.save(file, x_test)
        return x_train, df_train, x_val, df_val, x_test, df_test
    
    def LogisticRegression_train(self, dataloader):
        """
        Once embeddings are obtained, use logistic regression for evaluation
        """
        self.model.eval()
        #Get embeddings
        x = []
        y_true = []
        survival_data = []
        df = dataloader.iterable.dataset.df
        with torch.no_grad():
            for image, coords, text, clinical, gene_data, label, case_id in dataloader:
                metadata = df[df["case_id"]==case_id[0]]
                if len(clinical):
                    clinical = clinical.to(self.device)
                image, coords = image.to(self.device), coords.to(self.device)
                if isinstance(gene_data,dict):
                    gene_data = {key: value.to(self.device) for key, value in gene_data.items()}
                else:
                    gene_data = gene_data.to(self.device)
                
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    logit = self.multitask_forward(x=image, coords=coords, genes=gene_data, clinical=clinical, task_ids=[0])
                if isinstance(label,torch.Tensor):
                    label = label.tolist()
                y_true.append(str(label[0]))
                survival_data.append((metadata.iloc[0][["vital_status","durations"]].to_dict()))
                x.append(logit.ravel().cpu().numpy())
        
        #Diagnosis
        y_true = np.asarray(y_true,int).ravel()
        x = np.array(x)
        x_filter, y_true = self.filter_labelset(x, y_true)
        #Logistic Regression against standard labels
        self.lr_eval = LogisticRegression(random_state=0, max_iter=200, solver="liblinear")
        self.lr_eval.fit(x_filter,y_true)
        pred_train = self.lr_eval.predict(x_filter)
        pred_probs = self.lr_eval.predict_proba(x_filter)
        #survival
        self.cph = lifelines.fitters.coxph_fitter.CoxPHFitter(penalizer=0.1)
        df_survival_data = pd.DataFrame(survival_data)
        df_model = pd.DataFrame(x,columns=list(np.arange(x.shape[1]).astype(str)))
        df_model["durations"] = df_survival_data["durations"]
        df_model["vital_status"] = df_survival_data["vital_status"]
        df_model.dropna(inplace=True)
        self.cph.fit(df_model, duration_col='durations', event_col='vital_status')
        c_index = self.cph.score(df_model, scoring_method="concordance_index")
        return y_true.tolist(), pred_train.tolist(), pred_probs.tolist(), c_index

    def filter_labelset(self, x, y):
        """
        Only select indicies which has certain labels available
        By default while making dataset, labels with y=-1 consists of rare labels not included for classification
        """
        idx = np.where(y>=0)[0]
        x_new = x[idx,:]
        y_new = y[idx]
        return x_new, y_new
    
    def evaluate(self, dataloader, stage):
        """
        Evaulate weakly-supervised MIL
        :param dataloader: torch dataloader object to iterate through
        :param stage: either 'val' or 'test' based on the eval stage
        :return: a tuple of model outputs to compute metrics on
        """
        self.model.eval()
        
        with torch.no_grad():

            x_logits = []
            y_true = []
            survival_data = []
            total_loss = 0
            n_iters = 0
            df = dataloader.iterable.dataset.df

            for image, coords, text, clinical, gene_data, label, case_id in dataloader:
                if len(clinical):
                    clinical = clinical.to(self.device)
                image, text, coords = image.to(self.device), text.to(self.device), coords.to(self.device)
                metadata = df[df["case_id"]==case_id[0]]
                if isinstance(gene_data,dict):
                    gene_data = {key: value.to(self.device) for key, value in gene_data.items()}
                else:
                    gene_data = gene_data.to(self.device)
                text = text.squeeze(0)
                text = self.projector(text)
                text = text / text.norm(dim=-1, keepdim=True)

                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    #logit is prediction for the text prompt across different task prompts
                    logit = self.multitask_forward(x=image, coords=coords, genes=gene_data, clinical=clinical, task_ids=[0,1,2])
                    logit_change = logit / logit.norm(dim=-1, keepdim=True)
                    loss = self.loss_fn(nn.functional.log_softmax(logit_change / self.temperature, dim=1),
                                        nn.functional.softmax(text[[0,1,3],:] / self.temperature, dim=1),
                                        ) * (self.temperature ** 2) * 10 

                x_logits.append(logit[[0],:].ravel().cpu().numpy())
                if isinstance(label,torch.Tensor):
                    label = label.tolist()
                y_true.append(str(label[0]))
                survival_data.append((metadata.iloc[0][["vital_status","durations"]].to_dict()))
                total_loss += loss.item()
                n_iters += 1

            print(f"Validation loss: {total_loss / n_iters}")
            #NO-op if epochs not divisible by some specified constant
            if (stage=="val") and (self.current_epoch%self.args.eval_interval!=0):
                return None, None, None, total_loss / n_iters
            
            if (stage=="test"):
                # Need to get the linear models for the best model
                self.cph, self.lr_eval = None, None
                train_iter = tqdm(self.get_eval_iterator(self.train_data["data"]))
                _,_,_,_ = self.LogisticRegression_train(train_iter)
            
            x_logits = np.array(x_logits)
            x_logits_filter, y_true = self.filter_labelset(x_logits, np.asarray(y_true,int).ravel())
            
            y_pred = self.lr_eval.predict(x_logits_filter)
            y_probs = self.lr_eval.predict_proba(x_logits_filter)
            #survival
            df_survival_data = pd.DataFrame(survival_data)
            df_model = pd.DataFrame(x_logits,columns=list(np.arange(x_logits.shape[1]).astype(str)))
            df_model["durations"] = df_survival_data["durations"]
            df_model["vital_status"] = df_survival_data["vital_status"]
            df_model.dropna(inplace=True)
            c_index = self.cph.score(df_model, scoring_method="concordance_index")
            return y_true.tolist(), y_pred.tolist(), y_probs.tolist(), total_loss / n_iters, c_index

    def compute_metrics(self, outputs, stage):
        """
        Compute relevant metrics for slide-based weakly supervised learning to track progress
        :param outputs: tuple of model outputs to compute metrics on
        :param stage: stage of experiment denoted by a string
        :return: a dictionary of metrics to be tracked and a key metric used to choose best validation model
        """
        metrics_dict = {}
        y_true_wsi, y_pred_wsi, y_probs_wsi, cls_loss, c_index = outputs

        #NO-OP if inputs are none
        if y_true_wsi is None:
            return metrics_dict, -1.0
        
        if self.args.num_classes<2:
            average="binary"
        else:
            average=None

        # get general confusion matrix metrics
        acc_wsi = metrics.accuracy_score(y_true_wsi, y_pred_wsi)
        bal_acc_wsi = metrics.balanced_accuracy_score(y_true_wsi, y_pred_wsi)
        recall_wsi = metrics.recall_score(y_true_wsi, y_pred_wsi, average=average)
        precision_wsi = metrics.precision_score(y_true_wsi, y_pred_wsi, average=average)
        f1_wsi = metrics.f1_score(y_true_wsi, y_pred_wsi, average=average)

        # return metrics dict for wandb with confusion matrix + ROC
        metrics_dict.update({
            f'{stage}_cls_acc': acc_wsi,
            f'{stage}_bal_cls_acc': bal_acc_wsi,
            f'{stage}_c_index': c_index,
            f'{stage}_cls_loss': cls_loss,
            f'{stage}_cls_recall': recall_wsi,
            f'{stage}_cls_precision': precision_wsi,
            f'{stage}_cls_f1': f1_wsi,
            f'{stage}_cls_conf_matrix': wandb.plot.confusion_matrix(y_true=y_true_wsi, preds=y_pred_wsi),
            f'{stage}_cls_ROC_curve': wandb.plot.roc_curve(y_true=y_true_wsi, y_probas=y_probs_wsi, classes_to_plot=list(np.arange(self.args.num_classes)))
        })

        # accuracy as key metric
        key_metric = metrics_dict[f'{stage}_bal_cls_acc']
        return metrics_dict, key_metric

    def configure_wandb_metrics(self):
        """
        Configures wandb metrics for MIL training
        """
        wandb.config.run_task = self.run_task

        # other metrics
        for stage in ['train', 'val', 'test']:
            # metrics for all MIL
            wandb.define_metric(f"{stage}_c_index", summary="max")
            wandb.define_metric(f"{stage}_cls_loss", summary="min")
            wandb.define_metric(f"{stage}_cls_acc", summary="max")
            wandb.define_metric(f"{stage}_cls_bal_acc", summary="max")
            wandb.define_metric(f"{stage}_cls_recall", summary="max")
            wandb.define_metric(f"{stage}_cls_precision", summary="max")
            wandb.define_metric(f"{stage}_cls_f1", summary="max")

    def deploy_mil(self):
        """
        Deploys pretrained MIL model on a new test set
        1. Loads the best model weights
        2. Generates feature embeddings for each WSI
        3. Performs logistic regression for classification tasks and cox regression for survival tasks
        Additionally saves the feature embeddings and labels for future use
        """
        assert self.args.world_size <= 1, 'Distributed training not supported with this eval function!'

        self.init_model_and_optimizer()
        # self.Dataset_Class = partial(FeaturesGeneTextDataset, case_wise=True, return_case=True)
        self.Dataset_Class = partial(FeaturesGeneTextDataset, 
                                     case_wise=True, 
                                     return_case=True,
                                     gene_group_defination=self.gene_group_defination, 
                                     threshold=self.THRESHOLD)

        wandb.init(mode="disabled")
        
        #load dataset
        train_iter = self.get_eval_iterator(self.train_data["data"])
        val_iter = self.get_eval_iterator(self.val_data['data'])
        test_iter = self.get_eval_iterator(self.test_data['data'])
        
        #load model
        state_dict = torch.load(self.args.eval_weights,map_location="cpu")
        miss, unexpected = self.model.load_state_dict(state_dict)
        print(f'Testing with model saved weights: {self.args.eval_weights}\nMissed: {miss}\nUnexpected: {unexpected}')
        self.model.eval()
 
        label_set = ["vital_status","durations","primary_class","primary_diagnosis","ajcc_pathologic_stage","ajcc_pathologic_t","ajcc_pathologic_n","ajcc_pathologic_m","project_id"]
        x_train, df_train, x_val, df_val, x_test, df_test = self.get_features(train_iter, val_iter, test_iter, label_set)
        print("Feature vectors saved...")
        perform_testing(x_train, df_train, x_test, df_test, penalizer=0.1)

def run_trainer(args, modaltune_trainer):
    """
    Runs the MIL trainer with the given arguments
    Processes the arguments for different types of experiments
    :param args: argparse arguments entered for experiment
    :param modaltune_trainer: the trainer class to use for training
    """
    if args.eval_only:
        #For loading specific model more conviniently by direclty refering to config file
        config_path = str(Path(args.eval_weights).parent / "config.json")
        print(f"Loading configuration from {config_path}")
        with open(config_path, "r") as f:
            #temporary args
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
        #If train/test/val/other data are provided then use those instead
        #We assume this is OOD mode
        if args.train_json!="./train.json":
            #Model applied in OOD setting, changing data related information
            t_args.train_json = args.train_json
            t_args.val_json = args.val_json
            t_args.test_json = args.test_json
            t_args.text_location = args.text_location
            t_args.genomics_csv_path = args.genomics_csv_path
            t_args.clinical_location = args.clinical_location
            t_args.num_classes = args.num_classes
        t_args.eval_only = args.eval_only
        t_args.eval_weights = args.eval_weights
        t_args.eval_name = args.eval_name
        t_args.multi_seed = 0
        args = t_args

    #For easy multi-seed experiments
    if args.multi_seed:
        seeds = [args.seed, args.seed+1, args.seed+2]
    else:
        seeds = [args.seed]
    for seed in seeds:
        args.seed = seed
        trainer = modaltune_trainer(args)
        if not args.eval_only:
            key_metric = trainer.run()
            if args.save_embeddings:
                #Generate embeddings for the best model weights for future use
                trainer.args.eval_weights = str(Path(trainer.args.output_path) / "best_model_weights.pt")
                if args.eval_name is None:
                    strtime = time.strftime("%d%b_%H_%M_%S", time.localtime())
                    trainer.args.eval_name = args.model_config + f"_{strtime}"
                trainer.args.eval_name = args.eval_name + f"_seed_{seed}"
                trainer.deploy_mil()
        else:
            trainer.deploy_mil()

if __name__ == "__main__":
    # add new args here
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classification classes for evaluation')
    parser.add_argument('--model_config', default='conch_ViT-B-16_prompt', type=str,
                        help='model config for pretrained ViT')
    parser.add_argument('--mil_name',default="abmil", type=str,
                        help="choose between different aggregators")
    parser.add_argument('--text_location',default="./data/BRCA_textembeddings_conch_ViT-B-16_all.pt", type=str,
                        help="location of text embeddings")
    parser.add_argument('--gc', default=1, type=int,
                        help='number of gradient accumulation steps')
    parser.add_argument('--threshold', default=25000, type=int,
                        help='maximum number of patches to consider')
    parser.add_argument('--num_tasks',default=3, type=int,
                        help="Number of tasks to train the model on simultaneously")
    parser.add_argument('--genomics_csv_path',type=str,default="./data/genomics.csv",
                        help="Location for mRNA sequencing data in csv format")
    parser.add_argument('--clinical_location', default="", type=str,
                        help="location of simple clinical features")
    parser.add_argument("--save_embeddings", action="store_true", default=False,
                        help="save embeddings of best model weights after training")

    # args for evaluating on new data
    parser.add_argument('--eval_only', default=0, type=int,
                        help='set to 1 if looking to only eval on a new dataset with pretrained weights')
    parser.add_argument('--eval_weights', type=str,
                        help='path to ensemble model weights for only eval')
    parser.add_argument('--eval_name', type=str, default="mil",
                        help="name for saving feature embeddings for evaluation")
    
    args = parser.parse_args()
    if args.clinical_location.lower() in ["none","null","nan"]:
        args.clinical_location = ""

    #Run experiments
    modaltune_trainer = MILTextGeneTrainer_multitask
    run_trainer(args, modaltune_trainer)
