"""
Extracts patches in memory from a given whole slide image, available in form of torch dataset
"""

import torch
import numpy as np
import openslide
from pathlib import Path
from tqdm import tqdm
import torch.utils.data as data

from dplabtools.slides.patches import WholeImageGridPatches, MemPatchExtractor
from dplabtools.slides.processing import WSITissueMask
from dplabtools.slides import GenericSlide


class ExtractPatches(data.Dataset):
    def __init__(
        self,
        wsi_file: str,
        patch_size: int,
        level_or_mpp: float,
        foreground_threshold: float,
        patch_stride: int,
        mask_threshold: float,
        mask_kernelsize: int,
        remove_holes: bool = False,
        num_workers: int = 4,
        transform=None,
        tta_transform=None,
        save_preview=False,
        save_mask=False,
        output_dir="./",
        default_spacing=0.25,  # Set for TCGA, should be 0.25
        img_type="numpy",
    ) -> None:
        super().__init__()
        self.default_spacing = default_spacing
        self.img_type = img_type

        # Set default spacing for slides with no spacing information
        GenericSlide.set_external_mpp(default_spacing)

        self.transform = transform
        self.tta_transform = tta_transform

        self.scan = openslide.OpenSlide(wsi_file)
        iw, ih = self.scan.dimensions

        # Extract tissue masks around level with the given size
        level = self.find_best_level(size=1000)
        self.mask = WSITissueMask(
            wsi_file=wsi_file,
            level_or_minsize=level,
            mode="lab",
            color_threshold=mask_threshold,
            close_fill_kernel_size=mask_kernelsize,
            remove_small_holes_ratio=0,
            remove_small_objects_ratio=0,
            remove_all_holes=remove_holes,
        )
        # Extract coordinates of patches to be extracted
        self.grid_patches = WholeImageGridPatches(
            wsi_file=wsi_file,
            mask_data=self.mask.array,
            patch_size=patch_size,
            level_or_mpp=level_or_mpp,
            foreground_ratio=foreground_threshold,
            patch_stride=patch_stride,
            overlap_ratio=1.0,
            weak_label="label",
        )
        # Extractor class for patch extraction and saving it in memory
        self.extractor = MemPatchExtractor(
            patches=self.grid_patches,
            num_workers=num_workers,
            inference_mode=True,
            resampling_mode="tile",
        )

        if save_preview:
            self.grid_patches.save_preview_image(
                Path(output_dir) / f"preview_{Path(wsi_file).stem}.png"
            )

        if save_mask:
            self.mask.save_png(Path(output_dir) / f"mask_{Path(wsi_file).stem}.png")

        self.dataset_orig = []
        patch_sample = patch_size
        sh = sw = patch_sample
        ph = pw = patch_stride * patch_sample
        self.dataset_orig = [0] * len(self.grid_patches.patch_data)
        self.coords = [0] * len(self.grid_patches.patch_data)
        # Get x,y coordinates
        all_coords = np.array(
            [[points[0][1], points[0][0]] for points in self.grid_patches.patch_data]
        )
        max_height, max_width = np.max(all_coords, axis=0)
        temp_height = int(max((ih - ph), max_height) // sh + 1)
        temp_width = int(max((iw - pw), max_width) // sw + 1)
        # Get template for plotting
        self._template = np.zeros(shape=(temp_height, temp_width), dtype=np.float32)

        for patch in tqdm(self.extractor.patch_stream, desc="Patch Extraction"):
            patch_index = patch[2]
            if self.img_type == "numpy":
                img_tmp = np.asarray(patch[0])
            else:
                img_tmp = patch[0]
            self.dataset_orig[patch_index] = img_tmp
            self.coords[patch_index] = self.grid_patches.patch_data[patch_index][0]
            self._template[
                self.grid_patches.patch_data[patch_index][0][1] // sh,
                self.grid_patches.patch_data[patch_index][0][0] // sw,
            ] = (
                patch_index + 1
            )

    def __len__(self):
        return len(self.dataset_orig)

    def __getitem__(self, index):
        img = self.dataset_orig[index]
        coord = self.coords[index]
        if self.transform is not None:
            # For Test time augmentations
            if self.tta_transform is not None:
                img_tta = [self.transform(img)]
                for i in range(15):
                    img_tta.append(
                        self.transform(self.tta_transform(image=img)["image"])
                    )
                return torch.stack(img_tta, dim=0), coord
            else:
                return self.transform(img), coord
        else:
            return img, coord

    def find_best_level(self, size):
        level_dimensions = self.scan.level_dimensions
        for i, levels in enumerate(level_dimensions[::-1]):
            # Select level with dimension around 1000 otherwise it becomes too big
            if levels[0] > size or levels[1] > size:
                break
        return len(level_dimensions) - 1 - i

    @property
    def dataset(self):
        return self.dataset_orig

    @property
    def template(self):
        return self._template

    @property
    def coordinates(self):
        return self.coords
