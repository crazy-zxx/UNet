import copy
import os
from pathlib import Path

import numpy as np
import torch
from torch import optim, save
import SimpleITK as sitk

from torch.utils.data import DataLoader
from data.Hippocampus import Hippocampus, resize_image_itk
from model.unet3d import UNet
from utils.DiceLoss import DiceLoss
from utils.drawCurve import draw
from utils.oneHot import onehot2mask

train_datasets_path = r'./datasets/3d/hippocampus'
model_save_path = r'./saved_model_3d_hippocampus'
curve_save_path = r'./curve_3d_hippocampus'
val_save_path = r'./pred_3d_hippocampus'
batch_size = 1
n_classes = 3
epochs = 100


def train_val_split(ratio):
    """ 按照比率（ratio）将训练集分割成训练集和验证集 """
    # load train data
    h = Hippocampus(dirname=train_datasets_path, train=True)
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


def train():
    ratio = 0.3
    h_train, h_val = train_val_split(ratio)

    train_dataloader = DataLoader(h_train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(h_val, batch_size=1, shuffle=True, num_workers=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = UNet(n_channels=1, n_classes=n_classes).to(device)

    loss_func = DiceLoss()

    lr = 1e-2
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print('==== start train ====')

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

        model.eval()
        with torch.no_grad():
            total_acc = 0.
            steps = 0
            for i, (img, label) in enumerate(val_dataloader):
                img = img.to(device)
                label = label.to(device)
                pred = model(img)
                total_acc += 1 - loss_func(pred, label)
                steps += 1

                pred = pred.squeeze()
                # 通过onehot获取patch预测结果
                pred_image = onehot2mask(pred)

                itk_image = sitk.ReadImage(h_val.images[i])
                image_size = itk_image.GetSize()  # 读取该数据的size
                image_spacing = itk_image.GetSpacing()  # 读取该数据的spacing
                image_origin = itk_image.GetOrigin()
                image_direction = itk_image.GetDirection()

                pred_itk_img = sitk.GetImageFromArray(pred_image)
                pred_itk_img.SetSpacing(image_spacing)  # 设置spacing
                pred_itk_img.SetOrigin(image_origin)
                pred_itk_img.SetDirection(image_direction)
                pred_itk_img = resize_image_itk(pred_itk_img, image_size)
                # predicted image save path
                global val_save_path
                val_save_path = val_save_path + f'-epoch_{epoch}'
                os.makedirs(val_save_path, exist_ok=True)
                # Path(filepath).stem 从路径名中获取无扩展名(gz)的文件名
                pred_img_name = os.path.join(val_save_path, f'{Path(h_val.images[i]).stem}')
                # save image
                sitk.WriteImage(pred_itk_img, pred_img_name)
                print(f'save {pred_img_name} successfully!')

            val_avg_acc = total_acc / steps
            print(f'epoch:{epoch + 1}/{epochs} --> val acc:{val_avg_acc}')
            val_acc.append(val_avg_acc)

            if val_avg_acc >= max(val_acc):
                os.makedirs(model_save_path, exist_ok=True)
                model_name = os.path.join(model_save_path, 'best_model.pth')
                save(model.state_dict(), model_name)
                print(f'{model_name} , save best model successfully!')

        # 绘制损失、准确率图像并保存
        draw(epoch + 1, [train_loss, val_acc], ['train_loss', 'val_acc'], 'train-val', 'epoch', 'value',
             ['red', 'green'], curve_save_path)


if __name__ == '__main__':
    train()
