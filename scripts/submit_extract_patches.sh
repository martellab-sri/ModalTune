#!/bin/bash
cd ./data_utils

INPUT_DIR=PATHTODATABASE/
OUTPUT_DIR=PATHTODATABASE/TCGA-extractedfeatures/
ONCO_CODE=BRCA

python TCGA_extract_feats_TITAN.py --onco_code $ONCO_CODE --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR
# python TCGA_extract_feats_GIGAPATH.py --onco_code $ONCO_CODE --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR

