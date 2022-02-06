import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

class ImageDataset(Dataset):
    def __init__(self, image, bmi, transforms_=None, mode="train"): # before, after ==> root
        self.transform = transforms_
        self.image = image
        self.bmi = bmi

    def __getitem__(self, index):
        image_file = Image.open(self.image[index % len(self.image)])
        image_ya = self.transform(image_file)

        bmi_ya = torch.tensor(self.bmi[index % len(self.bmi)])

        return {"image":image_ya, "bmi":bmi_ya}

    def __len__(self):
        return len(self.image)