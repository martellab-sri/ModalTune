"""
Trainer class to assist in model training, validation and testing
"""

import random
import time
import shutil
import numpy as np
import wandb
import os
import json
from pathlib import Path
from torch.utils import data
import tqdm
import torch
import torch.distributed as dist


def infinite_loader(loader):
    """
    Create an infinite data loader
    :param loader: data loader
    :return: infinite data loader
    """
    while True:
        yield from loader


class Trainer:
    """Base trainer class"""

    def __init__(self, args):
        """
        Initialize the trainer
        :param args: argparse arguments entered for experiment
        """
        self.args = args

        # init distributed training if runtime environment is set up for it, otherwise do normal training
        self.init_distributed()

        # save arguments in output path
        if (self.args.rank == 0) and (self.args.eval_only == 0):
            # if not os.path.exists(self.args.output_path):
            #     os.makedirs(self.args.output_path)
            # else:
            #     raise Warning('Output path aready exists!')
            strtime = time.strftime("%d%b_%H_%M_%S", time.localtime())
            self.args.output_path = str(
                Path(self.args.output_path)
                / f"{self.args.mil_name}_exp_{strtime}_seed_{self.args.seed}"
            )
            try:
                Path.mkdir(Path(self.args.output_path), exist_ok=False, parents=True)
            except:
                # To avoid overwriting if multiple scripts are run at once
                self.args.output_path = (
                    self.args.output_path + f"_{np.random.randint(0,100000)}"
                )
                Path.mkdir(Path(self.args.output_path), exist_ok=False, parents=True)
            with open(f"{self.args.output_path}/config.json", "w") as fp:
                json.dump(vars(self.args), fp, indent=4)
                print(vars(self.args))
            # Copy model json to output path
            copy_location = Path(__file__).resolve().parent.parent / "model_configs"
            # search for file with that name
            filename = list(copy_location.glob(f"{self.args.model_config}.*"))[0]
            shutil.copyfile(str(filename), f"{self.args.output_path}/{filename.name}")

        # set seed
        self.set_seed(self.args.seed)
        # initialize device to use
        self.device = self.args.device

        # save once at end of training...
        if self.args.save_interval == -1:
            self.args.save_interval = self.args.num_epochs

        # empty inits (define in outer scope)
        self.Dataset_Class = None
        self.train_transforms = None
        self.eval_transforms = None
        self.model = None
        self.optimizer = None

        # inits to be defined when getting data
        self.train_folds = None
        self.val_folds = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

        # load datalists and save them to output folder
        self.do_val_test = True
        self.get_datalists()

        # flag for autolambda
        self.autolambda = False

    @staticmethod
    def set_seed(random_seed):
        # Initialize experiment
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)

    def init_model_and_optimizer(self):
        """
        Defines model and optimizer, implemented in inheritor class
        Needed for reinitializing model and optimizer for k-fold crossval
        """
        raise NotImplementedError

    def train_one_epoch(self, dataloader):
        """
        Defines one epoch of model training, implemented in inheritor class
        :param dataloader: torch dataloader object to iterate through
        :return: a tuple of model outputs to compute metrics on
        """
        raise NotImplementedError

    def evaluate(self, dataloader, stage):
        """
        Defines model evaluation on validation or test stages, implemented in inheritor class
        :param dataloader: torch dataloader object to iterate through
        :param stage: either 'val' or 'test' based on the eval stage
        :return: a tuple of model outputs to compute metrics on
        """
        raise NotImplementedError

    def validate(self, dataloader):
        """
        Wrapper to run validate code
        :param dataloader: torch dataloader object to iterate through
        :return: a tuple of model outputs to compute metrics on
        """
        return self.evaluate(dataloader, stage="val")

    def test(self, dataloader):
        """
        Wrapper to run test code
        :param dataloader: torch dataloader object to iterate through
        :return: a tuple of model outputs to compute metrics on
        """
        return self.evaluate(dataloader, stage="test")

    def compute_metrics(self, outputs, stage):
        """
        Compute relevant metrics to track progress
        :param outputs: tuple of model outputs to compute metrics on
        :param stage: stage of experiment denoted by a string
        :return: a dictionary of metrics to be tracked and a key metric used to choose best validation model
        """
        raise NotImplementedError

    def init_distributed(self):
        """
        Initializes distributed training if the running environment has been set up properly with environment vars
        (both slurm running or torch.distributed.launch initialization supported!)
        """
        # set WORLD_SIZE in environemnt when training with slurm, it's set automatically by torch.distributed.launch!
        if "WORLD_SIZE" in os.environ:
            self.args.world_size = int(os.environ["WORLD_SIZE"])
        else:
            self.args.world_size = 1

        if self.args.world_size > 1:
            # this arg is defined in torch.distributed.launch
            if self.args.local_rank is not None:
                self.args.rank = self.args.local_rank
                self.args.device = self.args.local_rank
                assert torch.cuda.device_count() == self.args.world_size, (
                    "set one GPU per process when using " "torch.distributed.launch!"
                )
            # this env variable is defined when running with slurm
            elif "SLURM_PROCID" in os.environ:
                self.args.rank = int(os.environ["SLURM_PROCID"])
                self.args.device = 0  # only permit 1 gpu per node
                assert (
                    torch.cuda.device_count() == 1
                ), "set one GPU per node in slurm for distributed training!"

            torch.cuda.set_device(self.args.device)
            print(
                f"CUDA device adjusted to device={self.args.device} for distributed training!"
            )
            # initialize process group
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=self.args.world_size,
                rank=self.args.rank,
            )
            print(
                f"Initialized (rank/world-size) {self.args.rank}/{self.args.world_size}"
            )
        else:
            self.args.rank = 0
            print("Distributed not available, using standard train!")

    def apply_ddp_to_model(self):
        """
        If doing distributed training, add DDP wrapper to model!
        """
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[self.args.device]
        )

    def get_datalists(self):
        """
        Grab train, val and test json files and log them in the experiment output path
        """
        # get train/val/test json files describing data
        with open(self.args.train_json, "r") as fp:
            self.train_data = json.load(fp)
        # test and val are not mandatory, but we will not validate or test or save "best model" in this case
        if self.args.val_json == "":
            assert (
                self.args.test_json == ""
            ), "Validation set is required to choose best model when doing test!"
            self.do_val_test = False
        else:
            with open(self.args.val_json, "r") as fp:
                self.val_data = json.load(fp)
            with open(self.args.test_json, "r") as fp:
                self.test_data = json.load(fp)

        # save them for logging purposes!
        if self.args.rank == 0:
            with open(f"{self.args.output_path}/train_datalist.json", "w") as fp:
                json.dump(self.train_data, fp, indent=4)
            if self.do_val_test:
                with open(f"{self.args.output_path}/val_datalist.json", "w") as fp:
                    json.dump(self.val_data, fp, indent=4)
                with open(f"{self.args.output_path}/test_datalist.json", "w") as fp:
                    json.dump(self.test_data, fp, indent=4)

    def get_kfolds(self):
        """
        Grab k-fold splits for training and validation and log them in the experiment output path
        """
        # if no fold json defined
        if self.args.fold_json is None:
            raise ValueError("If doing k-fold training, you need to define a fold json")

        # val and test must occur for k-fold training!
        self.do_val_test = True

        # otherwise...
        with open(self.args.fold_json, "r") as fp:
            fold_json = json.load(fp)
            train_folds = fold_json["train"]
            val_folds = fold_json["val"]

        assert self.args.num_folds == len(
            train_folds
        ), "train json does not define folds equal to num_folds"
        assert self.args.num_folds == len(
            val_folds
        ), "val json does not define folds equal to num_folds"

        self.train_folds = train_folds
        self.val_folds = val_folds

        # save k-fold info for logging
        if self.args.rank == 0:
            with open(f"{self.args.output_path}/kfold_splits.json", "w") as fp:
                json.dump(fold_json, fp)

    def get_train_iterator(self, datalist):
        """
        Get the iterator for train datalist using the dataset class defined in the inheritor trainer class
        :param datalist: list of items consisting the data to iterate through
        :return: torch dataloader object for training stage
        """
        train_ds = self.Dataset_Class(
            self.args, datalist, self.train_transforms, evaluate=False
        )
        if self.args.world_size > 1:
            train_sampler = data.distributed.DistributedSampler(train_ds, shuffle=True)
        else:
            train_sampler = None
        loader_params = {
            "batch_size": self.args.batch_size,
            "shuffle": (train_sampler is None),
            "pin_memory": True,
            "drop_last": self.args.drop_last,
            "num_workers": self.args.workers,
            "sampler": train_sampler,
        }
        return data.DataLoader(train_ds, **loader_params)

    def get_eval_iterator(self, datalist):
        """
        Get the iterator for eval datalist using the dataset class defined in the inheritor trainer class
        :param datalist: list of items consisting the data to iterate through
        :return: torch dataloader object for evaluation stages
        """
        eval_ds = self.Dataset_Class(
            self.args, datalist, self.eval_transforms, evaluate=True
        )
        if self.args.world_size > 1:
            eval_sampler = data.distributed.DistributedSampler(eval_ds, shuffle=False)
        else:
            eval_sampler = None
        loader_params = {
            "batch_size": self.args.batch_size,
            "shuffle": False,
            "pin_memory": True,
            "drop_last": self.args.drop_last,
            "num_workers": self.args.workers,
            "sampler": eval_sampler,
        }
        return data.DataLoader(eval_ds, **loader_params)

    def save_model(self, fold, epoch=None):
        """
        Saves model to output path, includes fold if required
        :param fold: fold number for trained model to save
        :param epoch: epoch of training weights
        """
        # saved every save_interval
        if epoch is not None:
            if fold is None:
                model_path = f"{self.args.output_path}/model_weights_epoch_{epoch}.pt"
            else:
                model_path = f"{self.args.output_path}/model_weights_epoch_{epoch}_fold_{fold}.pt"
        # saved if new best validation performance reached
        else:
            if fold is None:
                model_path = f"{self.args.output_path}/best_model_weights.pt"
            else:
                model_path = (
                    f"{self.args.output_path}/best_model_weights_fold_{fold}.pt"
                )
        torch.save(self.model.state_dict(), model_path)

    def load_best_model(self, fold):
        """
        Loads best model from output path, includes fold if required
        :param fold: fold number for trained model to load
        """
        if fold is None:
            model_path = f"{self.args.output_path}/best_model_weights.pt"
        else:
            model_path = f"{self.args.output_path}/best_model_weights_fold_{fold}.pt"
        pretrained_dict = torch.load(model_path)
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if pretrained_dict == {}:
            raise Warning("No model weights were loaded!")
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    def configure_wandb_metrics(self):
        """
        Configures and initializes wandb metrics, implemented in inheritor class
        """
        raise NotImplementedError

    @staticmethod
    def log_and_print_metrics(metrics, epoch):
        """
        Logs and prints metrics returned by training and evaluation loops
        :param metrics: metrics dictionary to log and print
        :param epoch: epoch of model trained
        """
        wandb.log(metrics, step=epoch)

        print(f"epoch: {epoch} || ", end="")
        for key in metrics:
            print(f"{key}: {metrics[key]} || ", end="")
        print()

    def gather_distributed_outputs(self, outputs):
        """
        If using distributed training, gathers model outputs from all ranks into rank 0 for computing total metrics
        :param outputs: outputs gathered onto rank=0 process, all other processes will output None
        :return:
        """
        # if not doing distributed, no need to gather
        if self.args.world_size <= 1:
            return outputs
        # otherwise, gather based on output type
        else:
            # objects will be gathered on rank=0
            if self.args.rank == 0:
                gathered_outputs = []
                for output in outputs:
                    # list to put gathered outputs into
                    gathered_output = [None for _ in range(self.args.world_size)]
                    dist.gather_object(output, gathered_output, dst=0)

                    # lists are concatenated together
                    if isinstance(output, list):
                        concat_list = []
                        for process_output in gathered_output:
                            concat_list.extend(process_output)
                        gathered_outputs.append(concat_list)

                    # numbers are averaged
                    elif isinstance(output, int) or isinstance(output, float):
                        averaged_val = np.mean(gathered_output)
                        gathered_outputs.append(averaged_val)

                    # no other model outputs should be accepted
                    else:
                        raise TypeError(
                            "Unexpected output type found when gathering outputs for distributed training!"
                        )
                return tuple(gathered_outputs)

            # other ranks still need the gather_object call, but you don't pass in the list for gathered outputs
            else:
                for output in outputs:
                    dist.gather_object(output, dst=0)
                return None

    def run(self):
        """
        Runs a training experiment, uses k-fold if specified in arguments
        """
        # run without kfolds
        if self.args.num_folds == 1:

            # initialize model and optimizer before ddp wrapper
            self.init_model_and_optimizer()

            # apply wrapper if doing ddp
            if self.args.world_size > 1:
                self.apply_ddp_to_model()

            # wandb log on node 0
            if self.args.rank == 0:
                run = wandb.init(config=self.args, mode=self.args.wandb_mode)
                self.configure_wandb_metrics()
                print("Using standard train")
                key_test_metric = self._run()
                run.finish()

            else:
                print("Using standard train")
                key_test_metric = self._run()

        # run with kfolds
        elif self.args.num_folds > 1:
            print("Using kfold crossval")
            if self.args.world_size > 1:
                raise ValueError(
                    "k-fold crossval not supported with distributed training!"
                )

            key_test_metric = self._run_kfold()
        return key_test_metric

    def _run(self, fold=None):
        """
        Runs a 1-fold (standard) training experiment
        """
        # get iterators for data folds
        if fold is None:
            train_iter = self.get_train_iterator(self.train_data["data"])
            if self.do_val_test:
                val_iter = self.get_eval_iterator(self.val_data["data"])
        else:
            train_iter = self.get_train_iterator(self.train_folds[fold])
            if self.do_val_test:
                val_iter = self.get_eval_iterator(self.val_folds[fold])

        # use a key metric to define best models
        key_test_metric = None
        best_val_metric = -np.Inf
        best_val_epoch = -1
        for epoch in range(self.args.num_epochs):
            self.current_epoch = epoch
            print(f"========== epoch: {epoch} ==========")

            # set epoch to adjust distributed sampling seed per epoch
            if self.args.world_size > 1:
                train_iter.sampler.set_epoch(epoch)
            if self.autolambda:
                # val for autolambda, use train iterator so it's random order
                infinite_val = infinite_loader(
                    self.get_train_iterator(self.val_data["data"])
                )
                train_outputs = self.train_one_epoch(
                    tqdm.tqdm(train_iter), infinite_val
                )
            else:
                train_outputs = self.train_one_epoch(tqdm.tqdm(train_iter))
            train_outputs = self.gather_distributed_outputs(train_outputs)
            if self.do_val_test:
                val_outputs = self.validate(tqdm.tqdm(val_iter))
                val_outputs = self.gather_distributed_outputs(val_outputs)

            # compute gathered metrics on rank=0 process
            if self.args.rank == 0:

                train_metrics, key_train_metric = self.compute_metrics(
                    train_outputs, "train"
                )
                self.log_and_print_metrics(train_metrics, epoch)

                # if validating, check for new best model
                if self.do_val_test:
                    val_metrics, key_val_metric = self.compute_metrics(
                        val_outputs, "val"
                    )
                    self.log_and_print_metrics(val_metrics, epoch)

                    if key_val_metric >= best_val_metric:
                        best_val_metric = key_val_metric
                        best_val_epoch = epoch
                        self.save_model(fold)
                        print(
                            f"new best model at epoch: {epoch} with val metric: {best_val_metric}"
                        )

                # also save every save_interval
                if (epoch + 1) % self.args.save_interval == 0:
                    self.save_model(fold, epoch=epoch)

        # run testing
        if self.do_val_test:
            test_iter = self.get_eval_iterator(self.test_data["data"])
            print(
                f"testing with model saved on epoch {best_val_epoch} on rank {self.args.rank}"
            )
            self.load_best_model(fold)
            test_outputs = self.test(tqdm.tqdm(test_iter))
            test_outputs = self.gather_distributed_outputs(test_outputs)

            if self.args.rank == 0:
                test_metrics, key_test_metric = self.compute_metrics(
                    test_outputs, "test"
                )
                self.log_and_print_metrics(test_metrics, self.args.num_epochs - 1)
                print(f"test key metric of: {key_test_metric}")
        return key_test_metric

    def _run_kfold(self):
        """
        Runs a k(>1) fold training experiment
        """
        # load in kfold json
        self.get_kfolds()
        run_name = None

        # run for each fold
        for fold in range(self.args.num_folds):

            # init run and model for fold
            run = wandb.init(config=self.args, reinit=True)
            self.init_model_and_optimizer()

            # set all base run names of folds to be the same!
            if fold == 0:
                run_name = run.name
            run.name = f"{run_name}_fold_{fold}"

            # configure metrics, add fold number to config file
            self.configure_wandb_metrics()
            run.config.fold_number = fold

            self._run(fold=fold)

            run.finish()
