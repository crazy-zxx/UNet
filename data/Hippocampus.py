import os

import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.oneHot import mask2onehot

classes_label = [0, 1, 2]


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int32)
    resampler.SetReferenceImage(itkimage)
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)

    return itkimgResampled


class Hippocampus(Dataset):
    def __init__(self, dirname, train=True):
        super(Hippocampus, self).__init__()

        # 所有图像resize到相同大小
        self.image_size = (64, 64, 64)

        self.train = train
        if self.train:
            self.path = f'{dirname}/train'
        else:
            self.path = f'{dirname}/test'

        # 训练数据集文件名列表
        self.images = []
        self.labels = []
        for f in os.listdir(self.path):
            if 'image' in f:
                for i in os.listdir(os.path.join(self.path, f)):
                    fn = os.path.join(self.path, f, i)
                    if os.path.isfile(fn):
                        self.images.append(fn)
                    else:
                        print('wrong')
            elif 'label' in f:
                for i in os.listdir(os.path.join(self.path, f)):
                    fn = os.path.join(self.path, f, i)
                    if os.path.isfile(fn):
                        self.labels.append(fn)
                    else:
                        print('wrong')
            else:
                print('not found right directory!')

    def __len__(self):
        return len(self.images)

    def normalization(self, data):
        """   数据归一化  """
        a = data - np.min(data)
        b = np.max(data) - np.min(data)
        # 避免除0操作
        return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b != 0)

    def __getitem__(self, item):
        if self.train:
            # 读取图像，并resize。数据shape会由image_size(x,y,z)变成(z,y,x) ！！！
            image = sitk.GetArrayFromImage(resize_image_itk(sitk.ReadImage(self.images[item]), self.image_size))
            label = sitk.GetArrayFromImage(resize_image_itk(sitk.ReadImage(self.labels[item]), self.image_size))
            # 图像归一化，标签不用归一化
            image = self.normalization(image)
            # label分割多通道（onehot）
            label = mask2onehot(label, classes_label)

            return torch.tensor(image).unsqueeze(dim=0).float(), torch.tensor(label).float()
        else:
            # 读取图像，并resize
            image = sitk.GetArrayFromImage(resize_image_itk(sitk.ReadImage(self.images[item]), self.image_size))
            # 图像归一化
            image = self.normalization(image)

            return torch.tensor(image).unsqueeze(dim=0).float()


if __name__ == '__main__':

    h = Hippocampus(dirname='../datasets/3d/hippocampus', train=True)
    batch_size = 3
    dataloader = DataLoader(h, batch_size=batch_size, shuffle=True, num_workers=0)
    for image, label in dataloader:
        print(image.shape, label.shape)
        break
