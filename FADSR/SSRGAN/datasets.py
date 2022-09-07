import glob
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm


def denormalize(tensors, mean, std):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape, scale, mean, std):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // scale, hr_height // scale), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)
        path = self.files[index % len(self.files)]

        return {"lr": img_lr, "hr": img_hr, "path": path}

    def __len__(self):
        return len(self.files)


def MeanStd(data_path):
    """
    print(MeanStd("../data/img_align_celeba/"))
    :param Dataset:
    :return:
    """
    print("Calculate the mean and variance of the training data.")
    files = os.listdir(data_path)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for file in tqdm(files):
        img = Image.open(data_path + file)
        img = transforms.ToTensor()(img)
        for c in range(3):
            mean[c] += img[c, :, :].mean()
            std[c] += img[c, :, :].std()
    mean.div_(len(files))
    std.div_(len(files))

    return np.array(list(mean.numpy())), np.array(list(std.numpy()))
