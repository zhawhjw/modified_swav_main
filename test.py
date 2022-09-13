import torch
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

path = "D:\\Data\\3D-FUTURE-model\\train"

label2index = {}


def helper():
    temp = []

    for root, dirs, files in os.walk(path, topdown=True):

        for f in files:

            if f.endswith(".jpg"):
                temp.append(root + "\\" + f)

    image_path = [t for t in temp]
    labels = [t.split("\\")[-2] for t in temp]

    counter = 0
    for label in labels:

        if label not in label2index:
            label2index[label] = counter
            counter += 1

    labels_index = [label2index[t.split("\\")[-2]] for t in temp]

    return image_path, labels_index


class CustomImageDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):

        image_paths, labels = helper()
        self.img_labels = labels
        self.img_dir = image_paths
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_dir[idx]
        image = read_image(img_path).float()
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        data = data / 255
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


if __name__ == "__main__":
    training_data = CustomImageDataset()
    train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)

    mean, std = get_mean_and_std(train_dataloader)

    print(mean)
    print(std)