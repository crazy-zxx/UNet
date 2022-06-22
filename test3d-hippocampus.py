import os
from pathlib import Path

import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader

from data.Hippocampus import Hippocampus, resize_image_itk
from model.unet3d import UNet
from utils.oneHot import onehot2mask

test_datasets_path = r'./datasets/3d/hippocampus'
model_path = r'./saved_model_3d_hippocampus/best_model.pth'
pred_save_path = r'./pred_3d_hippocampus'
n_classes = 3


def test():
    h_test = Hippocampus(dirname=test_datasets_path, train=False)
    test_dataloader = DataLoader(h_test, batch_size=1, shuffle=False, num_workers=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = UNet(n_channels=1, n_classes=n_classes).to(device)
    # 加载模型
    model.load_state_dict(torch.load(model_path))

    # enter eval mode
    model.eval()

    with torch.no_grad():
        for i, img in enumerate(test_dataloader):
            img = img.to(device)
            # convert data: shape 4-->3, cuda-->cpu, tensor-->numpy
            pred = model(img).squeeze().cpu().numpy()
            # 通过onehot获取patch预测结果
            pred_image = onehot2mask(pred)

            itk_image = sitk.ReadImage(h_test.images[i])
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
            os.makedirs(pred_save_path, exist_ok=True)
            # Path(filepath).stem 从路径名中获取无扩展名(gz)的文件名
            pred_img_name = os.path.join(pred_save_path, f'{Path(h_test.images[i]).stem}')
            # save image
            sitk.WriteImage(pred_itk_img, pred_img_name)
            print(f'save {pred_img_name} successfully!')


if __name__ == '__main__':
    test()
