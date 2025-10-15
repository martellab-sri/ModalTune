#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --cpus-per-task=16         # CPU cores/threds
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=50G                  # more workers (cpus) require more ram
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/vramanathan/Projects/PromptTune_ddp/logs/tcgaddp_%j.log

conda activate genetune

PROJECT_DIR=/home/vramanathan/PromptTune_ddp/
SEED=0

# ONCO_CODE="BRCA"
# ONCO_CODE="NSCLC"
# ONCO_CODE="GBMLGG"
# ONCO_CODE="RCC"
ONCO_CODE="PANCANCER"
# ONCO_CODE="COADREAD" #OOD Datasets
# ONCO_CODE="BLCA" #OOD Datasets


if [ "$ONCO_CODE" = "PANCANCER" ]; then
    SCRIPTNAME=${PROJECT_DIR}/train_modaltune_pancancer.py
    NUM_CLASSES=2,2,2,3
else
    SCRIPTNAME=${PROJECT_DIR}/train_modaltune.py
    if [ "$ONCO_CODE" = "RCC" ];
        NUM_CLASSES=3 #For RCC
    else
        NUM_CLASSES=2 #For rest of the cancer sites
    fi
fi

OUTPUT_DIR=/home/vramanathan/scratch/amgrp/prompttune_outputs
TEXT_LOCATION=/aippmdata/public/TCGA/TCGA-extractedtexts/${ONCO_CODE}_textembeddings_conch_ViT-B-16_all_v3.pt
GENE_LOCATION=/aippmdata/public/TCGA/TCGA-genomics/processed/tcga_${ONCO_CODE,,}_xena_clean_pathway.csv
CLIN_LOCATION=""

# MODEL_CONFIG=modaltune_titan_config.json
MODEL_CONFIG=modaltune_gigapath_config.json

SAVE_NAME=test_${MODEL_CONFIG}_${ONCO_CODE}

TYPE=gene
# TYPE=gene_clinical

# MIL_NAME=titan_${TYPE}_adapter
MIL_NAME=longnetvit_${TYPE}_adapter #provgigapath

# JSON_EXT=_titan #for titan
JSON_EXT="" #for provgigapath

MULTI_SEED=1
NUM_TASKS=3
LR=0.0001
# THRESHOLD=15000 #For TITAN
THRESHOLD=25000 #For Prov-Gigapath

python $SCRIPTNAME \
  --train_json ${PROJECT_DIR}/dataset/json_splits/tcga_${ONCO_CODE,,}/train_${ONCO_CODE,,}_cls_feat${JSON_EXT}.json\
  --val_json ${PROJECT_DIR}/dataset/json_splits/tcga_${ONCO_CODE,,}/val_${ONCO_CODE,,}_cls_feat${JSON_EXT}.json \
  --test_json ${PROJECT_DIR}/dataset/json_splits/tcga_${ONCO_CODE,,}/test_${ONCO_CODE,,}_cls_feat${JSON_EXT}.json \
  --output_path $OUTPUT_DIR \
  --num_folds 1 \
  --eval_only 0 \
  --device 0 \
  --workers 8 \
  --use_amp \
  --model_config $MODEL_CONFIG \
  --num_classes $NUM_CLASSES \
  --lr $LR \
  --num_epochs 15 \
  --weight_decay 0.0005 \
  --wandb_mode disabled \
  --save_interval 1 \
  --gc 1 \
  --mil_name $MIL_NAME \
  --text_location $TEXT_LOCATION \
  --eval_interval 1 \
  --num_tasks $NUM_TASKS \
  --genomics_csv_path $GENE_LOCATION \
  --clinical_location $CLIN_LOCATION \
  --eval_name $SAVE_NAME \
  --seed $SEED \
  --threshold $THRESHOLD \
  --multi_seed $MULTI_SEED