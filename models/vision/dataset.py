import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class BeautyVisionDataset(Dataset):
    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)

        # ----------------------------
        # CLEAN DATA INSIDE DATASET
        # ----------------------------
        self.df = self.df.dropna(
            subset=["monk_skin_tone_label_us", "fitzpatrick_label"]
        ).reset_index(drop=True)

        self.df["monk_skin_tone_label_us"] = self.df["monk_skin_tone_label_us"].astype(int)
        self.df["fitzpatrick_label"] = self.df["fitzpatrick_label"].astype(int)

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # ----------------------------
        # PRE-COMPUTE VALID IMAGE ROWS
        # ----------------------------
        self.valid_indices = []

        for idx, row in self.df.iterrows():
            for col in ["image_1_path", "image_2_path", "image_3_path"]:
                if pd.isna(row[col]):
                    continue
                if os.path.exists(row[col]):
                    self.valid_indices.append(idx)
                    break

        print(f"Valid image samples: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def _load_image(self, path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None

    def __getitem__(self, index):
        df_idx = self.valid_indices[index]
        row = self.df.iloc[df_idx]

        for col in ["image_1_path", "image_2_path", "image_3_path"]:
            if pd.isna(row[col]):
                continue

            if os.path.exists(row[col]):
                img = self._load_image(row[col])
                if img is not None:
                    img = self.transform(img)

                    labels = {
                        "monk_skin_tone": torch.tensor(row["monk_skin_tone_label_us"], dtype=torch.long),
                        "fitzpatrick": torch.tensor(row["fitzpatrick_label"], dtype=torch.long),
                    }

                    return img, labels

        raise RuntimeError("No valid image found (this should not happen)")
