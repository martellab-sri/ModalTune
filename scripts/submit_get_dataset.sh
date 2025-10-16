#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --cpus-per-task=20         # CPU cores/threds
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G                  # more workers (cpus) require more ram
#SBATCH --time=2-0:00:00
#SBATCH --output=/home/vramanathan/Projects/PromptTune_ddp/logs/tcgaddp_%j.log

conda activate genetune
cd ./data_utils

ONCO_CODE="BRCA"
#Genomics 
#consists of folder containing raw genomics data
RAW_GENE_DIR="/aippmdata/public/TCGA/TCGA-genomics/raw/"
OUTPUT_GENE_DIR="/aippmdata/public/TCGA/TCGA-genomics/processed/"
OUTPUT_CLIN_DIR="/aippmdata/public/TCGA/TCGA-extractedtexts"
RAW_IMG_DIR="/aippmdata/public/TCGA/"
FEAT_DIR="/aippmdata/public/TCGA/TCGA-extractedfeatures/ProvGigapath/"

### RUN THIS SCRIPT AFTER PATCH FEATURES HAVE BEEN EXTRACTED
#Make dataset
python make_gene_dataset.py --onco_code $ONCO_CODE --raw_data_dir $RAW_GENE_DIR --output_dir $OUTPUT_GENE_DIR
#Assumes path for patch feature vectors are already available
python make_dataset.py --onco_code $ONCO_CODE --img_input_dir $RAW_IMG_DIR --gene_input_dir $OUTPUT_GENE_DIR --feat_input_dir $FEAT_DIR
python make_clinical.py --onco_code $ONCO_CODE --output_dir $OUTPUT_CLIN_DIR
python make_textemb_conch.py --onco_code $ONCO_CODE --output_dir $OUTPUT_CLIN_DIR
