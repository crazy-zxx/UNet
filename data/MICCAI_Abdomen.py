import os

import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.oneHot import mask2onehot

classes_label = list(range(14))  # 0-13


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


class Abdomen(Dataset):
    def __init__(self, dirname, train=True):
        super(Abdomen, self).__init__()
        self.train = train
        if self.train:
            self.path = f'{dirname}/train'
        else:
            self.path = f'{dirname}/test'
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
        a = data - np.min(data)
        b = np.max(data) - np.min(data)
        return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b != 0)

    def __getitem__(self, item):
        if self.train:
            # 图像resize
            image = sitk.GetArrayFromImage(resize_image_itk(sitk.ReadImage(self.images[item]), (128, 512, 512)))
            label = sitk.GetArrayFromImage(resize_image_itk(sitk.ReadImage(self.labels[item]), (128, 512, 512)))
            # label图像分割多通道
            label = mask2onehot(label, classes_label)
            # 归一化
            image = self.normalization(image)
            label = self.normalization(label)
            return torch.tensor(image).unsqueeze(dim=0).float(), torch.tensor(label).float()
        else:
            image = sitk.GetArrayFromImage(resize_image_itk(sitk.ReadImage(self.images[item]), (128, 512, 512)))
            image = self.normalization(image)
            return torch.tensor(image).unsqueeze(dim=0).float()


if __name__ == '__main__':

    h = Abdomen(dirname=r'E:\datasets\RawData', train=True)
    batch_size = 2
    dataloader = DataLoader(h, batch_size=batch_size, shuffle=False, num_workers=0)
    for image, label in dataloader:
        print(image.shape, label.shape)
        break
