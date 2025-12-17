"""
Script for processing dataset considering multiple modalities such as clinical texts, transcriptomics, histopathology 
Very specific to the TCGA dataset
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter


class BaseDataset(torch.utils.data.Dataset):
    """
    Base dataset class for ModalTune multi-modal learning.
    
    Handles data loading, preprocessing, and class balancing for TCGA dataset.
    Supports both case-wise and sample-wise data organization with optional
    class balancing for training stability.
    
    Args:
        args: Configuration object containing dataset parameters
        datalist: List of data dictionaries with sample information
        transforms: Data transformation pipeline
        evaluate (bool): Whether dataset is used for evaluation (disables balancing)
        case_wise (bool): If True, organize data by case/patient ID
        filter (bool): Whether to apply label filtering
        filter_labelset (list): Specific labels to include when filtering
    
    Attributes:
        _GLOBAL_SEED (int): Fixed seed for reproducible class balancing
    """

    _GLOBAL_SEED = 12345

    def __init__(
        self,
        args,
        datalist,
        transforms,
        evaluate=False,
        case_wise=False,
        filter=False,
        filter_labelset: list = None,
    ):
        self.args = args
        self.transforms = transforms
        self.evaluate = evaluate
        self.case_wise = case_wise
        self.filter = filter
        self.filter_labelset = filter_labelset
        self.labelset = args.labelset

        # Set global random seed for reproducible class balancing across distributed processes
        self.global_random = random.Random()
        self.global_random.seed(self._GLOBAL_SEED)

        # Prepare labels and apply filtering if specified
        self.datalist = self.prepare_labels(datalist)
        
        # Create DataFrame for easy case-wise access
        self.df = pd.DataFrame.from_dict(self.datalist)
        if case_wise:
            self.case_ids = self.df["case_id"].unique()

    def __len__(self):
        """Return dataset length based on organization mode."""
        if self.case_wise:
            return len(self.case_ids)
        else:
            return len(self.datalist)

    def prepare_labels(self, datalist):
        """
        Prepares labels based on the given labelset changing the datalist
        Hardcoded preprocessing step for TCGA dataset
        """
        if self.filter:
            if (
                self.labelset == "ajcc_pathologic_stage"
            ):  # For stage prediction, results not incl. in main paper
                stage_mapper = {
                    "Stage_I": "low",
                    "Stage_II": "low",
                    "Stage_III": "high",
                    "Stage_IV": "high",
                    "Stage_X": "high",
                    "nan": "nan",
                }
                label_encoder = LabelEncoder().fit(self.filter_labelset)
                print(f"Class mappings: {label_encoder.classes_}")
            new_datalist = []
            label_list = []
            for data_dict in datalist:
                label = data_dict[self.labelset]
                if (
                    self.labelset == "ajcc_pathologic_stage"
                ):  # make it into Stage_I, Stage_II, Stage_III, Stage_IV etc.
                    label = (
                        str(label)
                        .replace("A", "")
                        .replace("B", "")
                        .replace("b", "")
                        .replace("C", "")
                        .replace("c", "")
                        .replace("D", "")
                        .replace("d", "")
                        .replace(" (i+)", "")
                        .replace(" (i-)", "")
                        .replace("m", "")
                        .replace("i", "")
                        .replace(" ", "_")
                    )
                    label = stage_mapper[label]
                    if label in self.filter_labelset:
                        label = label_encoder.transform([label])[0]
                        label_list.append(label)
                        data_dict[self.labelset] = label
                        new_datalist.append(data_dict)
                elif self.labelset == "primary_class":
                    if label > -1:
                        label_list.append(label)
                        data_dict[self.labelset] = label
                        new_datalist.append(data_dict)
                else:
                    raise NotImplementedError
            return new_datalist
        else:
            return datalist

    def transform_data(self, x):
        return self.transforms(x).float()

    def __getitem__(self, index):
        raise NotImplementedError


class FeaturesGeneTextDataset(BaseDataset):
    def __init__(
        self,
        args,
        datalist,
        transforms=None,
        evaluate=False,
        gene_group_defination={},
        case_wise=True,
        return_case=False,
        filter=False,
        filter_labelset: list = None,
        return_images=True,
        threshold: int = 25000,
    ):
        """
        Ensure that args has "genomics_csv_path" available
        Parameters:
            datalist: list of data in dictionary format
            is_group: Set True to divide genes into different groups for easier processing
            evaluate: Run in evaluation mode
            case_wise: Return data by combining slides belonging to the same patient
            return_case: Return case id with each data point
            mode: Choose between patches/features to load direct patches or precomputed-features
            threshold: Limit in the number of patches to load per case
            return_image: Set true to return images
        """
        super().__init__(
            args,
            datalist,
            transforms,
            evaluate=evaluate,
            case_wise=case_wise,
            filter=filter,
            filter_labelset=filter_labelset,
        )
        self.textembeddings = torch.load(self.args.text_location, map_location="cpu")
        self.return_case = return_case
        # gene processing
        self.gene_df = pd.read_csv(args.genomics_csv_path)
        self.gene_group_defination = gene_group_defination
        self.normalizer = StandardScaler().fit(self.gene_df.iloc[:, 1:].values)
        self.gene_df.iloc[:, 1:] = self.normalizer.transform(
            self.gene_df.iloc[:, 1:].values
        )

        self.threshold = threshold
        # Get intersection between cases from gene and clinical dataframe
        self.df = self.df.merge(
            self.gene_df["case_id"],
            left_on="case_submitter_id",
            right_on="case_id",
            suffixes=("", "_y"),
        )
        self.return_images = return_images
        if case_wise:
            self.case_ids = self.df["case_id"].unique()

        if self.args.clinical_location:
            self.clinicaldata = torch.load(
                self.args.clinical_location, map_location="cpu"
            )
        else:
            self.clinicaldata = None

    @property
    def return_scaler(self):
        return self.normalizer

    def __getitem__(self, index):
        if not self.case_wise:
            case_id = self.df.iloc[index]["case_id"]
            case_submitter_id = self.df.iloc[index]["case_submitter_id"]
            if self.return_images:
                features = torch.load(self.df.iloc[index]["features_path"])
                image = features["features"]
                coords = features["coords"]
            else:
                image, coords = [], []
            label = self.df.iloc[index][self.labelset]
        else:
            case_id = self.case_ids[index]
            metadata = self.df[self.df["case_id"] == case_id]
            case_submitter_id = metadata["case_submitter_id"].iloc[0]
            imgs = []
            coords = []
            offset = 0
            if self.return_images:
                # Contains offset added to the Y coordinates between multiple slides
                for i in range(len(metadata)):
                    features = torch.load((metadata["features_path"].iloc[i]))
                    imgs.append(features["features"])
                    coords.append(features["coords"] + torch.tensor([0, offset]))
                    # Take the maximum coordinate along y axis and add 1500 to ensure enough sense of seperation
                    offset = features["coords"].max(dim=0)[0][1].item() + 1500
                image = torch.cat(imgs)
                coords = torch.cat(coords)
            else:
                image, coords = [], []
            label = metadata.iloc[0][self.labelset]
            if isinstance(label, pd.Series):
                # label is in pandas series format, convert to dictionary
                label = label.to_dict()

        text = self.textembeddings[case_id]
        clinical = []
        if self.clinicaldata:
            clinical = self.clinicaldata[case_id].float()

        if len(self.gene_group_defination) > 0:
            gene_data_array = self.gene_df.loc[
                self.gene_df["case_id"] == case_submitter_id
            ]
            gene_data = {}
            for i in range(len(self.gene_group_defination)):
                gene_data[i] = torch.tensor(
                    np.asarray(
                        gene_data_array[self.gene_group_defination[i]].values,
                        np.float32,
                    )
                ).squeeze(0)
        else:
            gene_data = torch.tensor(
                np.asarray(
                    self.gene_df.loc[
                        self.gene_df["case_id"] == case_submitter_id
                    ].values[:, 1:],
                    np.float32,
                )
            )[0]
        if self.return_images:
            n_img = len(image)
            # To avoid out of memory issues, we limit to a specific amount of threshold
            if n_img > self.threshold:
                idx = torch.randperm(n_img)[: self.threshold]
                idx = idx.sort()[0]
                image = image[idx, :]
                coords = coords[idx, :]
        if self.return_case:
            return image, coords, text, clinical, gene_data, label, case_id
        else:
            return image, coords, text, clinical, gene_data, label
