
import csv
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from preprocessing import preprocess_image

LABELS = {"Oil": 0, "No_oil": 1, "Lookalike": 2}


class OilSpillDataset(Dataset):
    def __init__(self, manifest_path, split="train",
                 size=256, channels="both",
                 noise_filter="none", normalization="fixed"):

        # Save the preprocessing settings
        self.size = size
        self.channels = channels
        self.noise_filter = noise_filter
        self.normalization = normalization

        # Find the dataset folder
        self.data_root = Path(manifest_path).resolve().parent

        # Read only the requested split
        with open(manifest_path) as f:
            self.rows = [
                row for row in csv.DictReader(f)
                if row["split"] == split
            ]

    def __len__(self):
        # Number of images
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        # Prepare the image using our chosen settings
        image = preprocess_image(
            self.data_root / row["image"],
            size=self.size,
            channels=self.channels,
            noise_filter=self.noise_filter,
            normalization=self.normalization
        )

        # Convert the class name into a number
        label = np.int64(LABELS[row["label"]])

        return image, label
