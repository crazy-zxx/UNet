import os
from pathlib import Path

import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader

from data.Hippocampus import Hippocampus, resize_image_itk
from model.unet2d import UNet
from model.unet3d import UNet
from model.ZUNet import ZUNet
from utils.oneHot import onehot2mask

test_datasets_path = r'./datasets/3d/hippocampus'
model_path = r'./saved_model_3d_hippocampus/best_model.pth'
pred_save_path = r'./pred_3d_hippocampus'
n_channels = 1
n_classes = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# test datasets object
test_datasets = Hippocampus(dirname=test_datasets_path, train=False)


def test():
    test_dataloader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=0)
    # load model net
    model = UNet(n_channels=n_channels, n_classes=n_classes).to(device)
    # load model parameters
    model.load_state_dict(torch.load(model_path))

    print('==== start predict ====')
    # predict
    model.eval()
    with torch.no_grad():
        for i, img in enumerate(test_dataloader):
            img = img.to(device)
            pred = model(img).squeeze().cpu().numpy()
            pred_image = onehot2mask(pred)
            # get original info of nii
            itk_image = sitk.ReadImage(test_datasets.images[i])
            image_size = itk_image.GetSize()
            image_spacing = itk_image.GetSpacing()
            image_origin = itk_image.GetOrigin()
            image_direction = itk_image.GetDirection()
            # set info for predicted nii
            pred_itk_img = sitk.GetImageFromArray(pred_image)
            pred_itk_img.SetSpacing(image_spacing)  # 设置spacing
            pred_itk_img.SetOrigin(image_origin)
            pred_itk_img.SetDirection(image_direction)
            pred_itk_img = resize_image_itk(pred_itk_img, image_size)
            # save predicted nii
            os.makedirs(pred_save_path, exist_ok=True)
            # get original filename
            pred_img_name = os.path.join(pred_save_path, f'{Path(test_datasets.images[i]).stem}')
            # save image
            sitk.WriteImage(pred_itk_img, pred_img_name)
            print(f'save {pred_img_name} successfully!')


if __name__ == '__main__':
    test()
