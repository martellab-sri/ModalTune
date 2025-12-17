#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --cpus-per-task=16         # CPU cores/threds
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=50G                  # more workers (cpus) require more ram
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/vramanathan/Projects/PromptTune_ddp/logs/tcgaddp_%j.log

conda activate genetune

PROJECT_DIR=/home/vramanathan/Projects/ModalTune/
SEED=0

#ID Datasets
# ONCO_CODE="BRCA"
# ONCO_CODE="NSCLC"
# ONCO_CODE="GBMLGG"
# ONCO_CODE="RCC"
# ONCO_CODE="PANCANCER"
#OOD Datasets
# ONCO_CODE="BLCA" 
ONCO_CODE="COADREAD"

SCRIPTNAME=${PROJECT_DIR}/train_modaltune.py
if [ "$ONCO_CODE" = "RCC" ]; then
    NUM_CLASSES=3 #For RCC
else
    NUM_CLASSES=2 #For rest of the cancer sites
fi

TEXT_LOCATION=/aippmdata/public/TCGA/TCGA-extractedtexts/${ONCO_CODE}_textembeddings_conch_ViT-B-16_all_v3.pt
GENE_LOCATION=/aippmdata/public/TCGA/TCGA-genomics/processed/tcga_${ONCO_CODE,,}_xena_clean_pathway.csv
CLIN_LOCATION=None

# MODEL_CONFIG=modaltune_titan_config.json
MODEL_CONFIG=modaltune_gigapath_config
MODEL_WEIGHTS=/home/vramanathan/scratch/amgrp/prompttune_outputs/longnetvit_gene_adapter_exp_17Dec_14_36_39_seed_0/best_model_weights.pt

SAVE_NAME=deploy_${MODEL_CONFIG}_${ONCO_CODE}

# JSON_EXT=_titan #for titan
JSON_EXT="" #for provgigapath

python $SCRIPTNAME \
  --train_json ${PROJECT_DIR}/dataset/json_splits/tcga_${ONCO_CODE,,}/train_${ONCO_CODE,,}_cls_feat${JSON_EXT}.json\
  --val_json ${PROJECT_DIR}/dataset/json_splits/tcga_${ONCO_CODE,,}/val_${ONCO_CODE,,}_cls_feat${JSON_EXT}.json \
  --test_json ${PROJECT_DIR}/dataset/json_splits/tcga_${ONCO_CODE,,}/test_${ONCO_CODE,,}_cls_feat${JSON_EXT}.json \
  --eval_only 1 \
  --num_classes $NUM_CLASSES \
  --text_location $TEXT_LOCATION \
  --genomics_csv_path $GENE_LOCATION \
  --clinical_location $CLIN_LOCATION \
  --eval_name $SAVE_NAME \
  --seed $SEED \
  --eval_weights $MODEL_WEIGHTS