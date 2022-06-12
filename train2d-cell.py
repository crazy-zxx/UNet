import copy
import os

import numpy as np
import torch
from torch import optim, save
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import CrossEntropyLoss

from data.FruitFlyCell import FruitFlyCell
from model.unet2d import UNet
from utils.drawCurve import draw

train_datasets_path = r'./datasets/2d/cell'
model_save_path = r'./saved_model_2d_cell'
curve_save_path = r'./curve_2d_cell'
batch_size = 1
n_classes = 2
epochs = 100


def train_val_split(ratio):
    cell_transform = transforms.Compose([
        transforms.ToTensor()  # data-->[0,1]
    ])
    # load train data
    ds = FruitFlyCell(dirname=train_datasets_path, train=True, transform=cell_transform)
    length = len(ds)
    # random choice sample
    val_index = np.random.choice(range(length), int(length * ratio), replace=False)
    # copy object
    ds_train = copy.copy(ds)
    ds_val = copy.copy(ds)
    # clear list
    ds_train.images, ds_train.labels, ds_val.images, ds_val.labels = [], [], [], []
    # set sample
    for i in range(length):
        if i in val_index:
            ds_val.images.append(ds.images[i])
            ds_val.labels.append(ds.labels[i])
        else:
            ds_train.images.append(ds.images[i])
            ds_train.labels.append(ds.labels[i])
    del ds

    return ds_train, ds_val


def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)
    m2 = target.view(num, -1)
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def train():
    ratio = 0.3
    ds_train, ds_val = train_val_split(ratio)

    train_dataloader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(ds_val, batch_size=1, shuffle=True, num_workers=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = UNet(n_channels=1, n_classes=n_classes).to(device)

    loss_func = CrossEntropyLoss()

    lr = 1e-1
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print('==== start train ====')

    train_loss = []
    val_acc = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
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
            print(f'epoch:{epoch + 1}/{epochs} --> val acc:{val_avg_acc}')
            val_acc.append(val_avg_acc)

            if val_avg_acc >= max(val_acc):
                os.makedirs(model_save_path, exist_ok=True)
                save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))
                print('save best model successfully!')

        draw(epoch + 1, [train_loss, val_acc], ['train_loss', 'val_acc'], 'train-val-2d', 'epoch', 'value',
             ['red', 'green'], curve_save_path)


if __name__ == '__main__':
    train()
