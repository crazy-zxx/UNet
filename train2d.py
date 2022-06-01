import copy
import os

import numpy as np
import torch
from torch import optim, save
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import MSELoss

from data.FruitFlyCell import FruitFlyCell
from model.unet2d import UNet
from utils.DrawCurve import draw


def train_val_split(ratio):
    cell_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # load train data
    ffcell = FruitFlyCell(dirname='datasets/2d', train=True, transform=cell_transform)
    length = len(ffcell)
    # random choice sample
    val_index = np.random.choice(range(length), int(length * ratio), replace=False)
    # copy object
    ffcell_train = copy.copy(ffcell)
    ffcell_val = copy.copy(ffcell)
    # clear list
    ffcell_train.images, ffcell_train.labels, ffcell_val.images, ffcell_val.labels = [], [], [], []
    # set sample
    for i in range(length):
        if i in val_index:
            ffcell_val.images.append(ffcell.images[i])
            ffcell_val.labels.append(ffcell.labels[i])
        else:
            ffcell_train.images.append(ffcell.images[i])
            ffcell_train.labels.append(ffcell.labels[i])
    del ffcell

    return ffcell_train, ffcell_val


def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)
    m2 = target.view(num, -1)
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def train():
    ratio = 0.3
    ffcell_train, ffcell_val = train_val_split(ratio)

    batch_size = 1
    train_dataloader = DataLoader(ffcell_train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(ffcell_val, batch_size=1, shuffle=True, num_workers=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet(n_channels=1, n_classes=1).to(device)

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
            save_path = './saved_model_2d/'
            if val_avg_acc >= max(val_acc):
                os.makedirs('./saved_model', exist_ok=True)
                save(model.state_dict(), save_path+'best_model.pth')
                print('save best model successfully!')

        draw(epoch+1, [train_loss, val_acc], 'train-val', 'epoch', 'value', ['red', 'green'], './curve')


if __name__ == '__main__':
    train()
