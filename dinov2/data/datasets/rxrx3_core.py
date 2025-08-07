import csv
from enum import Enum
import logging
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from .extended import ExtendedVisionDataset

from datasets import load_dataset

from huggingface_hub import hf_hub_download
import pandas as pd


# Dummy decoders (replace with actual logic or imports)
class ImageDataDecoder:
    def __init__(self, data: bytes):
        self.data = data

    def decode(self):
        return Image.open(io.BytesIO(self.data)).convert("RGB")

# Actual dataset class
class RXRX3_CORE(ExtendedVisionDataset):
    def __init__(
        self,
        *,
        split: "RXRX3_CORE.Split",
        root: str,
        extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,):
        super().__init__(root=root, transforms=transform)

        # Get the dataset and it's path to metadata
        file_path_metadata = hf_hub_download("recursionpharma/rxrx3-core", filename="metadata_rxrx3_core.csv", repo_type="dataset")
        rxrx3_core_metadata = pd.read_csv(file_path_metadata)

        # Mapping between metadata and image paths
        rxrx3_core_metadata["mapping"] = rxrx3_core_metadata.experiment_name + "/Plate" + rxrx3_core_metadata.plate.astype(str) + "/" + rxrx3_core_metadata.address.astype(str) 
        rxrx3_core_metadata["unique_mapping"] = rxrx3_core_metadata.treatment + "_" + rxrx3_core_metadata.concentration.astype(str)

        # Find all unique classes and create a mapping to integer ids
        unique_classes = rxrx3_core_metadata["unique_mapping"].unique()
        unique_classes.sort()
        self.class_mapping = {k: i for i, k in enumerate(unique_classes)}

        # Create a mapping from image paths to their metadata
        self.image_set = load_dataset("recursionpharma/rxrx3-core", split='train')
        mapping = pd.DataFrame(self.image_set["__key__"]).reset_index().rename(columns={0: "paths", "index": "id"})

        self.metadata = rxrx3_core_metadata
        self.mapping = mapping

    def __len__(self):
        return len(self.metadata)

    def get_target(self, index: int):
        # Get the target name from metadata and map it to an integer id
        target_name = self.metadata.iloc[index]["unique_mapping"]
        target_id = self.class_mapping[target_name]
        return target_id 

    def __getitem__(self, index: int):

        # try:

        id_ = self.metadata.iloc[index]["mapping"]

        target = self.get_target(index)

        id_index = int(self.mapping[self.mapping["paths"] == id_+"_s1_1"]["id"].values[0])
        img_p = np.array(self.image_set[id_index:id_index+5]["jp2"])
        img_p = Image.fromarray(np.transpose(img_p, (1, 2, 0)), 'RGB')

        # except Exception as e:
        #     raise RuntimeError(f"can not read image for sample {index} ------ {id_} ------ {id_index}") from e

        print("PRE size: ", img_p.shape)

        if self.transforms is not None:
            img_p = self.transforms(img_p)

        print("POST size: ", img_p.size)

        return img_p, target