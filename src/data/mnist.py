import numpy as np
import torch
import os
import torchvision
from torch.utils.data import Dataset, Sampler, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.transforms import v2

from src.utils import instantiate_from_config


class MNIST(Dataset):
    #reminder to change root to train with MNIST
    def __init__(self, root = "/scratch/anji/data/MNIST", train = True, transform_fns = None):
        self.train = train
        if self.train:
            train_dataset = torchvision.datasets.MNIST(root = root, train = True, download = True)
            self.data = train_dataset.data.reshape(60000, 28*28)
        else:
            test_dataset = torchvision.datasets.MNIST(root = root, train = False, download = True)
            self.data = test_dataset.data.reshape(10000, 28*28)
                
        self.length = self.data.size(0)

        if transform_fns is not None:
            transforms = []
            for transform_fn in transform_fns:
                transforms.append(instantiate_from_config(transform_fn))
            self.transforms = v2.Compose(transforms)
        else:
            self.transforms = lambda x: x

    def __getitem__(self, index):
        img = self.data[index].long()

        return self.transforms(img)

    def __len__(self):
        return self.length
