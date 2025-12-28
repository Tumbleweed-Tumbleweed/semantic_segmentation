#dataset.py
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]

class PascalDataSet(Dataset):
    def __init__(self, root="PASCAL_Segmentation", names=None, image_size=256, training=False):
        self.image_size = image_size
        self.training = training

        root_path = REPO_ROOT / root
        image_folder = root_path / "Images"
        ann_folder = root_path / "Annotations"

        self.images = [str(image_folder / n) for n in names]
        self.labels = [str(ann_folder / n.replace(".jpg", ".png")) for n in names]

        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3,1,1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = Image.open(self.images[idx]).convert("RGB").resize((self.image_size, self.image_size))
        y = Image.open(self.labels[idx]).resize((self.image_size, self.image_size), Image.NEAREST)

        if self.training:
            if random.random() < 0.5:
                x = x.transpose(Image.FLIP_LEFT_RIGHT)
                y = y.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                x = x.transpose(Image.FLIP_TOP_BOTTOM)
                y = y.transpose(Image.FLIP_TOP_BOTTOM)

        x = torch.from_numpy(np.array(x, np.float32) / 255.0).permute(2,0,1)
        x = (x - self.mean) / self.std

        y = torch.from_numpy(np.array(y, np.uint8)).long()
        y[(y > 20) & (y != 255)] = 255
        return x, y
