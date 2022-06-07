import copy
import os
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import torch
from torch import optim, save
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from data.Hippocampus import Hippocampus
from model.unet3d import UNet
from utils.DrawCurve import draw


def train_val_split(ratio):

    # load train data
    h = Hippocampus(dirname='datasets/3d', train=True)
    length = len(h)
    # random choice sample
    val_index = np.random.choice(range(length), int(length * ratio), replace=False)
    # copy object
    h_train = copy.copy(h)
    h_val = copy.copy(h)
    # clear list
    h_train.images, h_train.labels, h_val.images, h_val.labels = [], [], [], []
    # set sample
    for i in range(length):
        if i in val_index:
            h_val.images.append(h.images[i])
            h_val.labels.append(h.labels[i])
        else:
            h_train.images.append(h.images[i])
            h_train.labels.append(h.labels[i])
    del h

    return h_train, h_val


def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)
    m2 = target.view(num, -1)
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def train():
    ratio = 0.3
    h_train, h_val = train_val_split(ratio)

    batch_size = 1
    train_dataloader = DataLoader(h_train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(h_val, batch_size=1, shuffle=True, num_workers=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_classes = 2
    model = UNet(n_channels=1, n_classes=n_classes).to(device)

    mse_loss = MSELoss()

    lr = 1e-2
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print('==== start train ====')
    epochs = 100
    train_loss = []
    val_acc = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        steps = 0
        for img, label in train_dataloader:
            img, label = img.to(device), label.to(device)
            pred = model(img)
            loss = mse_loss(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().cpu()
            steps += 1
        train_avg_loss = total_loss / steps
        print(f'epoch:{epoch} --> train loss:{train_avg_loss}')
        train_loss.append(train_avg_loss)

        model.eval()
        with torch.no_grad():
            total_acc = 0
            steps = 0
            for img, label in val_dataloader:
                img = img.to(device)
                pred = model(img).cpu()
                dice_acc = dice_coeff(pred=pred, target=label)
                total_acc += dice_acc
                steps += 1

            val_avg_acc = total_acc / steps
            print(f'epoch:{epoch} --> val acc:{val_avg_acc}')
            val_acc.append(val_avg_acc)
            save_path = './saved_model_3d/'
            if val_avg_acc >= max(val_acc):
                os.makedirs(save_path, exist_ok=True)
                save(model.state_dict(), save_path+'best_model.pth')
                print('save best model successfully!')

        draw(epoch + 1, [train_loss, val_acc], 'train-val', 'epoch', 'value', ['red', 'green'], './curve')


if __name__ == '__main__':
    train()