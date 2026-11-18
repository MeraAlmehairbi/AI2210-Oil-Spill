import csv
from pathlib import Path
import numpy as np
from preprocessing import preprocess_image

LABELS = {"Oil": 0, "No_oil": 1, "Lookalike": 2}

class OilSpillDataset:
    def __init__(self, manifest_path, split="train"):
        self.data_root = Path(manifest_path).resolve().parent
        with open(manifest_path) as f:
            self.rows = [
                row for row in csv.DictReader(f)
                if row["split"] == split
            ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = preprocess_image(self.data_root / row["image"])
        label = np.int64(LABELS[row["label"]])
        return image, label
