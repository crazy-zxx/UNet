import os
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.Hippocampus import Hippocampus
from model.unet3d import UNet


def test():
    n_classes = 2
    h_test = Hippocampus(dirname='datasets/3d', train=False)

    # must set 1
    batch_size = 1
    test_dataloader = DataLoader(h_test, batch_size=batch_size, shuffle=False, num_workers=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = r'./saved_model/best_model.pth'
    model = UNet(n_channels=1, n_classes=n_classes).to(device)
    model.load_state_dict(torch.load(model_path))

    # enter eval mode
    model.eval()

    with torch.no_grad():
        for i, img in enumerate(test_dataloader):
            img = img.to(device)
            # convert data: shape 4-->3, cuda-->cpu, tensor-->numpy
            pred = model(img).squeeze().cpu().numpy()
            # normalization and rescale to gray (0-255)
            for j in range(n_classes):
                pred[j] = ((pred[j] - np.min(pred[j])) / (np.max(pred[j]) - np.min(pred[j]))) * (j + 1)
            # transpose and convert to uint8
            pred = np.around(pred).astype(np.uint8)
            pred_img = sitk.GetImageFromArray(np.max(pred, axis=0))
            # predicted image save path
            pred_save_path = './pred'
            os.makedirs(pred_save_path, exist_ok=True)
            # Path(filepath).stem 从路径名中获取无扩展名的文件名
            pred_img_name = os.path.join(pred_save_path, f'{Path(h_test.images[i]).stem}')
            # save image
            # imageio.imwrite(pred_img_name, pred)
            sitk.WriteImage(pred_img, pred_img_name)
            print(f'save {pred_img_name} successfully!')


if __name__ == '__main__':
    test()
