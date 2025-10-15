# ModalTune: Fine-Tuning Slide-Level Foundation Models with Multi-Modal Information for Multi-task Learning in Digital Pathology
Code and data for the manuscript: [ICCV 2025] ModalTune: Fine-Tuning Slide-Level Foundation Models with Multi-Modal Information for Multi-task Learning in Digital Pathology. 
Linked preprint: https://arxiv.org/abs/2503.17564

## Description

## Key Features

## Experimental Results

## Project Structure
```
ModalTune/
├── train_modaltune.py              # Main training script for single cancer types
├── train_modaltune_pancancer.py    # Pan-cancer multi-task training script
│
├── data_utils/                     # Data processing and preparation
│
├── dataset/                        # Processed datasets and splits
│
├── models/                         # Model architectures and components
│   ├── aggregators/                # MIL aggregation modules including ModalAdapters
│   ├── genomic_utils/              # Genomic data processing
│   ├── vitadapter/                 # Vision adapter utils
│   └── prov_gigapath/              # Prov-GigaPath model components
│
├── model_configs/                  # Model configuration files
│
├── utils/                          # Utility functions and helpers
│   ├── constants.py                # Constants and paths
│   ├── test_utils_modaltune.py     # Evaluation utilities
│   └── test_utils_pancancer.py     # Pan-cancer evaluation utilities
│
└── scripts/                        # Execution scripts
    ├── deploy_OOD_modaltune.sh     # Out-of-distribution evaluation
    ├── submit_extract_patches.sh   # Patch extraction pipeline
    ├── submit_get_dataset.sh       # Dataset creation pipeline
    └── submit_modaltune.sh         # Training pipeline
```

## 📋 Requirements

### Python Dependencies

```bash
# Core ML Libraries
torch==2.0.0
torchvision==0.15.0

# Deep Learning & Vision
timm==1.0.7
transformers==4.36.2

# Additional
warmup_scheduler
gene_thesaurus #gene processing
dplabtools #patch extraction
conch #conch and conch related packages (https://github.com/mahmoodlab/CONCH)
prov_gigapath #For running longnetvit with flash attention and other dependencies (https://github.com/prov-gigapath/prov-gigapath)
```

## Running

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd ModalTune

# Create conda environment
conda create -n modaltune python=3.8
conda activate modaltune

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt

# Install CONCH (if using CONCH features)
pip install git+https://github.com/mahmoodlab/CONCH.git
```

### 2. Data Acquisition

### TCGA Histopathology Images

Download TCGA whole slide images along with clinical data from the GDC Data Portal (https://portal.gdc.cancer.gov/)

### Genomic Data (UCSC Xena Database)

Download genomic data from UCSC Xena Browser.

Pan-Cancer dataset was downloaded from [here](https://xenabrowser.net/datapages/?cohort=TCGA%20Pan-Cancer%20(PANCAN)&removeHub=http%3A%2F%2F127.0.0.1%3A7222) under the Gene expression RNAseq section.

For other individual cancer specific RNA seq, can be downloaded by navigating TCGA Hub in the Xena database.  (https://xenabrowser.net/datapages/).  


### 3. Pre-trained Model Weights
We used the following foundation models in our experiments. 
- **CONCH** (for text embeddings): https://github.com/mahmoodlab/CONCH
- **TITAN** (for ModalTune TITAN): https://huggingface.co/MahmoodLab/TITAN
- **Prov-GigaPath** (for ModalTune Gigapath): https://github.com/prov-gigapath/prov-gigapath

At the end ensure `utils/constants.py` has the correct paths set for your data directories.

### 4. Processing Raw Dataset

### Extract Patch Features
Patch features can be extracted using either TITAN or Prov-GigaPath models. Use the respective line in `scripts/submit_extract_patches.sh` to extract features.

```bash
bash scripts/submit_extract_patches.sh
```
### Process rest of the data
Use the scripts in `scripts/submit_get_dataset.sh` to process genomic data, clinical text and create dataset splits.
```bash
bash scripts/submit_get_dataset.sh
```

## Training ModalTune

#### Single Cancer Type Training

Ensure that the paths in `scripts/submit_modaltune.sh` are correctly set for your dataset and model (prov-gigapath/titan) configuration. Then run for each cancer type:
```bash
bash scripts/submit_modaltune.sh
```

#### Pan-Cancer Multi-Task Training
For pan-cancer, ensure you have the pancancer dataset splits created and modify `scripts/submit_modaltune.sh` to point to the pancancer config and dataset. Then run:
```bash
bash scripts/submit_modaltune.sh
```

### Evaluation Only (Pre-trained Model)
For out-of-distribution evaluation on a new test set, modify the paths and OOD cancer types in `scripts/deploy_OOD_modaltune.sh` and run:
```bash
bash scripts/deploy_OOD_modaltune.sh
```

## 🧠 Model Architecture

ModalTune implements a multi-modal architecture with:

1. **Vision Encoder**: Pre-trained models (CONCH/TITAN/Prov-GigaPath)
2. **Gene Encoder**: Pathway-based genomic feature encoding
3. **Text Embeddings**: Clinical text embeddings via CONCH
4. **Cross-Modal Fusion**: Adapter modules for multi-modal integration
5. **MIL Aggregation**: Multiple Instance Learning for slide-level prediction

## Acknowledgments

We would like to express our gratitude to the following projects and resources that have significantly contributed to the development of ModalTune.
### Genomic Data Processing
- **SurvPath**: https://github.com/mahmoodlab/SurvPath/

### Foundation Models
- **CONCH**: https://github.com/mahmoodlab/CONCH
- **TITAN**: https://huggingface.co/MahmoodLab/TITAN
- **Prov-GigaPath**: https://github.com/prov-gigapath/prov-gigapath

### Architecture Components
- **ViT-Adapter**: https://github.com/czczup/ViT-Adapter
- **Mask2Former**: https://github.com/facebookresearch/Mask2Former
- **MLP-Mixer**: https://github.com/lucidrains/mlp-mixer-pytorch
- **PromptKD**: https://github.com/zhengli97/PromptKD
- **timm**: https://github.com/rwightman/pytorch-image-models

### Data Sources
- **TCGA**: [The Cancer Genome Atlas](https://portal.gdc.cancer.gov/)
- **UCSC Xena**: [UCSC Xena Browser](https://xenabrowser.net/)

## 📝 Citation

If you use ModalTune in your research, please cite:

```bibtex
@article{ramanathan2025modaltune,
  title={ModalTune: Fine-Tuning Slide-Level Foundation Models with Multi-Modal Information for Multi-task Learning in Digital Pathology},
  author={Ramanathan, Vishwesh and Xu, Tony and Pati, Pushpak and Ahmed, Faruk and Goubran, Maged and Martel, Anne L},
  journal={arXiv preprint arXiv:2503.17564},
  year={2025}
}
```

## Contact

You can reach the authors by raising an issue in this repo or email them at
vishwesh.ramanathan@mail.utoronto.ca/tonylt.xu@mail.utoronto.ca/a.martel@mail.utoronto.ca
