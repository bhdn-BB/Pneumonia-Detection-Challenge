import pydicom
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset as BaseDataset
import os


class PneumoniaDataset(BaseDataset):

    def __init__(
            self,
            df,
            img_dir,
            transform=None
    ) -> None:
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(
            self.img_dir,
            f"{row.patientId}.dcm"
        )
        dcm = pydicom.dcmread(img_path)
        image = dcm.pixel_array.astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        boxes = row[
                   ['x', 'y', 'width', 'height']
                                               ].values.astype(np.float32)
        target = row['Target']
        if self.transform:
            image = self.transform(image)
        return (image,
                torch.tensor(target, dtype=torch.float32),
                torch.tensor(boxes, dtype=torch.float32))
