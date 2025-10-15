"""
Script for generating bag of feature vectors per slide for MIL training on TCGA dataset for Prov-Gigapath (https://github.com/prov-gigapath/prov-gigapath)
"""

import sys
import os
from pathlib import Path
import argparse

sys.path.append(Path(__file__).resolve().parent.parent)

import timm
import torch
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm

from utils.extract_patches import ExtractPatches

DEVICE = torch.device("cuda:0")

# Ensure that you set HF cache and HF token properly
print("Loading model")
model_fv = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).to(
    DEVICE
)
print("Loaded model")
transform_fv = transforms.Compose(
    [
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

#############################################################################################################################################################

TILE_SIZE = 256
TILE_STRIDE_SIZE = 1
LEVEL_OR_MPP = 0.5

parser = argparse.ArgumentParser(description="Configurations for extraction of ")
parser.add_argument("--onco_code", type=str, default="brca", help="type project code")
parser.add_argument(
    "--input_dir",
    default="./raw_data",
    type=str,
    help="directory containing of raw input files",
)
parser.add_argument(
    "--output_dir",
    default="./data",
    type=str,
    help="directory for storing processed files",
)
args = parser.parse_args()
ONCO_CODE = args.onco_code.upper

INPUT_DIR = list(Path(args.input_dir) / f"TCGA-{ONCO_CODE}/images/").rglob("*.svs")
OUTPUT_DIR = (
    Path(args.output_dir) / f"ProvGigapath/TCGA_MIL_Patches_GigaPath_1MPP_{ONCO_CODE}"
)
Path.mkdir(OUTPUT_DIR, parents=False, exist_ok=True)
processed_files = [files.stem.split("_")[0] for files in OUTPUT_DIR.glob("*.pt")]

for paths in INPUT_DIR:
    slide_name = paths.stem.split(".")[0]
    print(f"Processing {slide_name}...")
    if slide_name in processed_files:
        print("Already processed...")
        continue
    try:
        patch_dataset = ExtractPatches(
            wsi_file=str(paths),
            patch_size=TILE_SIZE,
            level_or_mpp=LEVEL_OR_MPP,
            foreground_threshold=0.95,
            patch_stride=TILE_STRIDE_SIZE,
            mask_threshold=0.1,
            mask_kernelsize=9,
            remove_holes=False,
            num_workers=30,
            save_preview=False,
            save_mask=False,
            transform=transform_fv,
            img_type="pillow",
        )
    except:
        print("Might be empty slide file")
        continue

    dataloader = torch.utils.data.DataLoader(
        patch_dataset, batch_size=512, num_workers=8
    )
    all_feats = []
    all_coords = []
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Extracting and saving feature vectors"):
            with torch.cuda.amp.autocast(enabled=True):
                img = data[0].to(DEVICE)
                feats = model_fv(img)
            all_feats.extend(feats.cpu())
            all_coords.extend(torch.stack(data[1], dim=1))
        all_feats = torch.stack(all_feats, dim=0)
        all_coords = torch.stack(all_coords, dim=0)
    print("Extracted {} features".format(len(all_feats)))
    torch.save(
        {"features": all_feats, "coords": all_coords},
        str(OUTPUT_DIR / f"{slide_name}_featvec.pt"),
    )
print("Done")
