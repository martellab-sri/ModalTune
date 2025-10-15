"""
Script for generating feature vectors per slide both on the patch and slide level for TITAN model (https://github.com/mahmoodlab/TITAN/tree/main)
"""

import sys
from pathlib import Path
import argparse

sys.path.append(Path(__file__).resolve().parent.parent)

import torch
from pathlib import Path
from tqdm import tqdm

from utils.extract_patches import ExtractPatches
from transformers import AutoModel

DEVICE = torch.device("cuda:0")

TILE_SIZE = 512
PATCHSIZE_LVL0 = 1024  # 0.25mpp

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
ONCO_CODE = args.onco_code.upper()

TILE_STRIDE_SIZE = 1
MPP = 0.5  # need to be floating point

print(
    f"Given settings \nTile size: {TILE_SIZE}\nTile stride size: {TILE_STRIDE_SIZE}\nMPP: {MPP}"
)

titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
conch, preprocess = titan.return_conch()  # 448x448 -> normalize
titan.to(DEVICE)
conch.to(DEVICE)
titan.eval()
conch.eval()

INPUT_DIR = list(Path(args.input_dir) / f"TCGA-{ONCO_CODE}/images/").rglob("*.svs")
OUTPUT_DIR = (
    Path(args.output_dir) / f"TITAN/TCGA_MIL_Patches_CONCHv1.5_{MPP}MPP_{ONCO_CODE}"
)
SLIDE_EMB_DIR = Path(args.output_dir) / f"TITAN/TCGA_SLIDEEMB_TITAN_{ONCO_CODE}"

Path.mkdir(OUTPUT_DIR, parents=False, exist_ok=True)
Path.mkdir(SLIDE_EMB_DIR, parents=False, exist_ok=True)

processed_files = [files.stem.split("_")[0] for files in OUTPUT_DIR.glob("*.pt")]

for paths in INPUT_DIR:
    slide_name = paths.stem.split(".")[0]
    print(f"Processing {slide_name}...")
    if slide_name in processed_files:
        print("Already processed...")
        continue
    try:
        patch_dataset = ExtractPatches(
            wsi_file=paths,
            patch_size=TILE_SIZE,
            level_or_mpp=MPP,
            foreground_threshold=0.95,
            patch_stride=TILE_STRIDE_SIZE,
            mask_threshold=0.1,
            mask_kernelsize=9,
            remove_holes=False,
            num_workers=16,
            save_preview=False,
            save_mask=False,
            transform=preprocess,
            tta_transform=None,
            default_spacing=0.25,
            img_type="pillow",
        )
    except Exception as e:
        print(f"{e}, Might be due to empty slide file")
        continue

    dataloader = torch.utils.data.DataLoader(
        patch_dataset, batch_size=512, num_workers=16
    )
    all_feats = []
    all_coords = []
    try:
        with torch.no_grad():
            for data in tqdm(dataloader, desc="Extracting and saving feature vectors"):
                img = data[0].to(DEVICE)
                feats = conch(img)
                all_feats.extend(feats.cpu())
                all_coords.extend(torch.stack(data[1], dim=1))
            all_feats = torch.stack(all_feats, dim=0)
            all_coords = torch.stack(all_coords, dim=0)

        print("Extracted {} Patch features".format(len(all_feats)))
        torch.save(
            {"features": all_feats, "coords": all_coords},
            str(OUTPUT_DIR / f"{slide_name}_featvec.pt"),
        )

        # Extracting slide embeddings
        # extract slide embedding
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ), torch.inference_mode():
            features = all_feats.unsqueeze(0).to(DEVICE)
            coords = all_coords.unsqueeze(0).to(DEVICE)
            slide_embedding = titan.encode_slide_from_patch_features(
                features, coords, PATCHSIZE_LVL0
            )

        print("Extracted Slide features".format(len(all_feats)))
        torch.save(
            slide_embedding.to(torch.float32),
            str(SLIDE_EMB_DIR / f"{slide_name}_featvec.pt"),
        )
    except:
        pass
print("Done")
