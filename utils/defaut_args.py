import argparse


parser = argparse.ArgumentParser()

# ======= Setup parameters =======

parser.add_argument("--device", default=0, type=int, help="device to train on")
parser.add_argument(
    "--use_amp", action="store_true", default=False, help="use mixed precision"
)
parser.add_argument(
    "--wandb_mode",
    default="disabled",
    type=str,
    help="wandb mode, choose out of online/offline/disabled. By default its disabled",
)
parser.add_argument("--seed", default=0, type=int, help="random seed")
parser.add_argument(
    "--multi_seed",
    default=0,
    type=int,
    help="set to 1 if repeating experiments across multiple seeds",
)

# ======= Training parameters =======

parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
parser.add_argument(
    "--weight_decay",
    default=0.01,
    type=float,
    help="weight decay/weights regularizer for sgd",
)
parser.add_argument(
    "--beta1", default=0.9, type=float, help="momentum for sgd, beta1 for adam"
)
parser.add_argument(
    "--beta2", default=0.999, type=float, help="momentum for sgd, beta1 for adam"
)

parser.add_argument("--num_epochs", default=10, type=int, help="epochs to train for")
parser.add_argument(
    "--eval_interval",
    default=1,
    type=int,
    help="Intervals at which the model should be evaluated, due to the cost of evaluation",
)
parser.add_argument(
    "--save_interval", default=-1, type=int, help="epoch interval to save model weights"
)
parser.add_argument(
    "--num_folds",
    default=5,
    type=int,
    help="number of folds to use for k-fold cross_val",
)
parser.add_argument(
    "--labelset", 
    default='primary_class',
    type=str,
    help='type of labels to use for training')

# ======= Dataloader parameters =======

parser.add_argument("--batch_size", default=128, type=int, help="input batch size")
parser.add_argument(
    "--workers",
    default=8,
    type=int,
    help="number of workers to use for GenerateIterator",
)
parser.add_argument(
    "--drop_last",
    default=False,
    type=lambda x: bool(int(x)),
    help="drop last batch if it is smaller than batch_size",
)
parser.add_argument(
    "--train_json", default="./train.json", type=str, help="json file for training data"
)
parser.add_argument(
    "--val_json", default="./val.json", type=str, help="json file for validation data"
)
parser.add_argument(
    "--test_json", default="./test.json", type=str, help="json file for testing data"
)
parser.add_argument(
    "--fold_json",
    default=None,
    type=str,
    help="json file detailing folds generated for k-fold crossval",
)

# ======= Output parameters =======

parser.add_argument(
    "--output_path",
    default="./results",
    type=str,
    help="path to output run logs and details",
)

# ======= Distributed training parameters =======

parser.add_argument(
    "--local-rank",
    default=None,
    type=int,
    help="argument defined in torch.distributed.launch during distributed training",
)
