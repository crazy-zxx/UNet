import os
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.FruitFlyCell import FruitFlyCell
from model.unet2d import UNet
from utils.oneHot import onehot2mask

test_datasets_path = r'datasets/2d/cell'
model_path = r'./saved_model_2d/best_model.pth'
pred_save_path = './pred2d'
n_classes = 2


def test():
    cell_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    ds_test = FruitFlyCell(dirname=test_datasets_path, train=False, transform=cell_transform)
    test_dataloader = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = UNet(n_channels=1, n_classes=n_classes).to(device)
    model.load_state_dict(torch.load(model_path))

    # enter eval mode
    model.eval()

    with torch.no_grad():
        for i, img in enumerate(test_dataloader):
            img = img.to(device)
            # convert data: shape 4-->3, cuda-->cpu, tensor-->numpy
            pred = model(img).squeeze().cpu().numpy()
            pred_img = onehot2mask(pred) * 255

            os.makedirs(pred_save_path, exist_ok=True)
            # Path(filepath).stem 从路径名中获取无扩展名的文件名
            pred_img_name = os.path.join(pred_save_path, f'{Path(ds_test.images[i]).stem}.png')
            # save image
            cv2.imwrite(pred_img_name, pred_img)
            print(f'save {pred_img_name} successfully!')


if __name__ == '__main__':
    test()
