import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, image_dirs, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dirs = image_dirs
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _find_image_path(self, image_id):
        filename = f"{image_id}.jpg"
        for directory in self.image_dirs:
            candidate = os.path.join(directory, filename)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"{filename} not found in {self.image_dirs}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self._find_image_path(row["image_id"])
        image = Image.open(image_path).convert("RGB")
        label = CLASS_TO_IDX[row["dx"]]

        if self.transform:
            image = self.transform(image)

        return image, label


def load_metadata(metadata_csv):
    return pd.read_csv(metadata_csv)
