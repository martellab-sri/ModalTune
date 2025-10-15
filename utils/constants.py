"""
Constants and configuration values used across the ModalTune repository.

This module centralizes all hardcoded values, file paths, and mappings
to ensure consistency and easy maintenance across the codebase.
"""

import os

# =============================================================================
# Model Weight Locations
# =============================================================================

# Gigapath model weights location
GIGAPATH_WEIGHT_LOC = "/aippmdata/trained_models/Martel_lab/pathology/huggingface/hub/models--prov-gigapath--prov-gigapath/"

# CONCH (Conversational Oncology CHatbot) model configuration
CONCH_CFG = "conch_ViT-B-16"
CONCH_CHECKPOINT_PATH = "/aippmdata/trained_models/Martel_lab/pathology/SSL_CONCH/CONCH/checkpoints/conch/conch.pt"

# TITAN model configuration
TITAN_CODE_PATH = "/aippmdata/trained_models/Martel_lab/pathology/huggingface/models/models--MahmoodLab--TITAN/snapshots/"
TITAN_SNAPSHOT_ID = "b2fb4f475256eb67c6e9ccbf2d6c9c3f25f20791"

# =============================================================================
# Dataset Mappings and Labels
# =============================================================================

# TCGA project ID mappings - maps combined cancer types to constituent projects
PROJECT_ID_MAP = {
    "TCGA-BLCA": ["TCGA-BLCA"],  # Bladder Urothelial Carcinoma
    "TCGA-BRCA": ["TCGA-BRCA"],  # Breast Invasive Carcinoma
    "TCGA-COADREAD": ["TCGA-COAD", "TCGA-READ"],  # Colorectal Adenocarcinoma
    "TCGA-GBMLGG": ["TCGA-GBM", "TCGA-LGG"],  # Brain Lower Grade Glioma + Glioblastoma
    "TCGA-NSCLC": ["TCGA-LUAD", "TCGA-LUSC"],  # Non-Small Cell Lung Cancer
    "TCGA-RCC": ["TCGA-KICH", "TCGA-KIRC", "TCGA-KIRP"],  # Renal Cell Carcinoma
    "TCGA-UCEC": ["TCGA-UCEC"],  # Uterine Corpus Endometrial Carcinoma
}

# =============================================================================
# Task Configuration
# =============================================================================

# Task type identifiers for multi-task learning
TASK_IDS = {
    0: "General",     # General cancer classification
    1: "Diagnosis",   # Diagnostic classification
    2: "Survival"     # Survival prediction
}

# =============================================================================
# Pan-Cancer Experiment Configuration
# =============================================================================

# Number of distinct cancer sites for pan-cancer analysis
NUM_SITES = 4

# Site label mapping for consistent pan-cancer labeling
# Maps TCGA project codes to numeric site labels
SITE_LABEL = {
    "TCGA-BRCA": 0,  # Breast cancer
    "TCGA-GBM": 1,   # Glioblastoma
    "TCGA-LGG": 1,   # Lower grade glioma (grouped with GBM)
    "TCGA-LUAD": 2,  # Lung adenocarcinoma  
    "TCGA-LUSC": 2,  # Lung squamous cell carcinoma (grouped with LUAD)
    "TCGA-KICH": 3,  # Kidney chromophobe
    "TCGA-KIRC": 3,  # Kidney renal clear cell carcinoma
    "TCGA-KIRP": 3,  # Kidney renal papillary cell carcinoma
}
