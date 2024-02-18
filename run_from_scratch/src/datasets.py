import torchvision
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
import pandas as pd
from sklearn import metrics
from src import utils as ut
from torch.utils.data import Dataset
import tqdm


def get_dataset(dataset_name, split, datadir):
    train_flag = True if split == 'train' else False
    if dataset_name == "mnist2d":
        dataset = torchvision.datasets.MNIST(datadir, train=train_flag,
                               download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.5,), (0.5,))
                               ])
                               )
    return DatasetWrapper(dataset, split=split)


class DatasetWrapper:
    def __init__(self, dataset, split):
        self.dataset = dataset
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]

        return {"images": data,
                'labels': target,
                'meta': {'indices': index}}