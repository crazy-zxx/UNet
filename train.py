import copy
import os

import numpy as np
import torch
from torch import optim, save

from torch.utils.data import DataLoader
from data.Hippocampus import Hippocampus
from model.unet3d import UNet
from utils.loss import DiceLoss, DiceBCELoss
from utils.drawCurve import draw

train_datasets_path = r'./datasets/3d/hippocampus'
model_save_path = r'./saved_model_3d_hippocampus'
curve_save_path = r'./curve_3d_hippocampus'
batch_size = 1
n_channels = 1
n_classes = 3
lr = 1e-2
epochs = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# split train datasets to train and val by ratio
ratio = 0.3
# train datasets object
train_val_datasets = Hippocampus(dirname=train_datasets_path, train=True)


def train_val_split(ratio):
    """ split the train datasets into train datasets and val datasets according to ratio """

    global train_val_datasets
    length = len(train_val_datasets)
    # random choice sample
    val_index = np.random.choice(range(length), int(length * ratio), replace=False)
    # copy object
    train_datasets = copy.copy(train_val_datasets)
    val_datasets = copy.copy(train_val_datasets)
    # clear list
    train_datasets.images, train_datasets.labels, val_datasets.images, val_datasets.labels = [], [], [], []
    # split
    for i in range(length):
        if i in val_index:
            val_datasets.images.append(train_val_datasets.images[i])
            val_datasets.labels.append(train_val_datasets.labels[i])
        else:
            train_datasets.images.append(train_val_datasets.images[i])
            train_datasets.labels.append(train_val_datasets.labels[i])
    del train_val_datasets

    return train_datasets, val_datasets


def train():

    train_datasets, val_datasets = train_val_split(ratio)
    # dataloader
    train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_datasets, batch_size=1, shuffle=True, num_workers=0)
    # model
    model = UNet(n_channels=n_channels, n_classes=n_classes).to(device)
    # loss function
    loss_func = DiceBCELoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print('==== start train ====')
    # train
    train_loss = []
    val_acc = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.
        steps = 0
        for img, label in train_dataloader:
            img, label = img.to(device), label.to(device)
            pred = model(img)
            loss = loss_func(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().cpu()
            steps += 1
        train_avg_loss = total_loss / steps
        print(f'epoch:{epoch + 1}/{epochs} --> train loss:{train_avg_loss}')
        train_loss.append(train_avg_loss)

        # val
        model.eval()
        with torch.no_grad():
            total_acc = 0.
            steps = 0
            for img, label in val_dataloader:
                img = img.to(device)
                pred = model(img).cpu()
                total_acc += 1 - loss_func(pred, label)
                steps += 1
            val_avg_acc = total_acc / steps
            print(f'epoch:{epoch + 1}/{epochs} --> val acc:{val_avg_acc}')
            val_acc.append(val_avg_acc)
            # save better model
            if val_avg_acc >= max(val_acc):
                os.makedirs(model_save_path, exist_ok=True)
                model_name = os.path.join(model_save_path, 'best_model.pth')
                save(model.state_dict(), model_name)
                print(f'{model_name} , save best model successfully!')

        # draw and save: loss and accuracy curve image
        draw(epoch + 1, [train_loss, val_acc], ['train_loss', 'val_acc'], 'train-val', 'epoch', 'value',
             ['red', 'green'], curve_save_path)


if __name__ == '__main__':
    train()
