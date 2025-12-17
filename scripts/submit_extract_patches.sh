#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --cpus-per-task=16        # CPU cores/threds
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G                  # more workers (cpus) require more ram
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/vramanathan/Projects/PromptTune_ddp/logs/tcgaddp_%j.log


conda activate promptune
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7

cd ./data_utils

INPUT_DIR=/aippmdata/public/TCGA/
OUTPUT_DIR=/aippmdata/public/TCGA/TCGA-extractedfeatures/
ONCO_CODE=BRCA

python TCGA_extract_feats_TITAN.py --onco_code $ONCO_CODE --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR
# python TCGA_extract_feats_GIGAPATH.py --onco_code $ONCO_CODE --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR

