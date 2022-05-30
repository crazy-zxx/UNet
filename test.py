import os
from pathlib import Path

import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.FruitFlyCell import FruitFlyCell
from model.unet2d import UNet


def test():
    cell_transform = transforms.Compose([
        transforms.ToTensor()  # data-->[0,1]
    ])
    ffcell_test = FruitFlyCell(dirname='datasets/2d', train=False, transform=cell_transform)

    # must set 1
    batch_size = 1
    test_dataloader = DataLoader(ffcell_test, batch_size=batch_size, shuffle=False, num_workers=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = r'./saved_model/best_model.pth'
    model = UNet(n_channels=1, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path))

    # enter eval mode
    model.eval()

    with torch.no_grad():
        for i, img in enumerate(test_dataloader):
            img = img.to(device)
            # convert data: shape 4-->3, cuda-->cpu, tensor-->numpy
            pred = model(img).squeeze().cpu().numpy()
            # normalization and rescale to gray (0-255)
            pred = ((pred - np.min(pred)) / (np.max(pred) - np.min(pred)))*255
            # transpose and convert to uint8
            pred = pred.astype(np.uint8)
            # predicted image save path
            pred_save_path = './pred'
            os.makedirs(pred_save_path, exist_ok=True)
            # Path(filepath).stem 从路径名中获取无扩展名的文件名
            pred_img_name = os.path.join(pred_save_path, f'{Path(ffcell_test.images[i]).stem}.png')
            # save image
            imageio.imwrite(pred_img_name, pred)
            print(f'save {pred_img_name} successfully!')


if __name__ == '__main__':
    test()
