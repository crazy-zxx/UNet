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

    def transform(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    def __getitem__(self, item):
        if self.train:
            # 图像resize
            image = sitk.GetArrayFromImage(resize_image_itk(sitk.ReadImage(self.images[item]), (64, 64, 64)))
            label = sitk.GetArrayFromImage(resize_image_itk(sitk.ReadImage(self.labels[item]), (64, 64, 64)))
            # label图像分割多通道
            label = mask2onehot(label, classes_label)
            # 归一化
            image = self.transform(image)
            label = self.transform(label)
            return torch.tensor(image).unsqueeze(dim=0).float(), torch.tensor(label).float()
        else:
            image = sitk.GetArrayFromImage(resize_image_itk(sitk.ReadImage(self.images[item]), (64, 64, 64)))
            image = self.transform(image)
            return torch.tensor(image).unsqueeze(dim=0).float()


if __name__ == '__main__':

    h = Hippocampus(dirname='../datasets/3d/hippocampus', train=True)
    batch_size = 1
    dataloader = DataLoader(h, batch_size=batch_size, shuffle=False, num_workers=0)
    for i, img in enumerate(dataloader):
        print(i)
