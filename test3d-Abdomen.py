import os
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.MICCAI_Abdomen import Abdomen, resize_image_itk
from model.unet3d import UNet
from utils.oneHot import onehot2mask

# test_datasets_path = r'./datasets/3d/Abdomen'
test_datasets_path = r'E:\datasets\Abdomen'
model_path = r'./saved_model_Abdomen/best_model.pth'
pred_save_path = r'./pred3d_Abdomen'
n_classes = 14


def test():
    h_test = Abdomen(dirname=test_datasets_path, train=False)
    test_dataloader = DataLoader(h_test, batch_size=1, shuffle=False, num_workers=0)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    model = UNet(n_channels=1, n_classes=n_classes).to(device)
    model.load_state_dict(torch.load(model_path))

    # enter eval mode
    model.eval()

    per_image_patchs = int(len(h_test) / len(h_test.images))
    with torch.no_grad():
        pred_image_patchs = []
        for i, img in enumerate(test_dataloader):
            img = img.to(device)
            # convert data: shape 4-->3, cuda-->cpu, tensor-->numpy
            pred = model(img).squeeze().cpu().numpy()
            pred_image_patchs.append(onehot2mask(pred))
            if (i + 1) % per_image_patchs == 0:
                itk_image = sitk.ReadImage(h_test.images[i // per_image_patchs])
                image_size = itk_image.GetSize()  # 读取该数据的size
                spacing = itk_image.GetSpacing()  # 读取该数据的spacing
                pred_image = np.reshape(np.hstack(pred_image_patchs), (128, 512, 512))
                pred_itk_img = resize_image_itk(sitk.GetImageFromArray(pred_image), image_size)
                # predicted image save path
                os.makedirs(pred_save_path, exist_ok=True)
                # Path(filepath).stem 从路径名中获取无扩展名的文件名
                pred_img_name = os.path.join(pred_save_path, f'{Path(h_test.images[i // per_image_patchs]).stem}')
                # save image
                pred_itk_img.SetSpacing(spacing) # 设置spacing，这一步别忘了
                sitk.WriteImage(pred_itk_img, pred_img_name)
                print(f'save {pred_img_name} successfully!')
                pred_image_patchs.clear()


if __name__ == '__main__':
    test()
