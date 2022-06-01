import os

import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)
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
            label1 = sitk.GetArrayFromImage(resize_image_itk(sitk.ReadImage(self.labels[item]), (64, 64, 64)))
            # label图像分割多通道
            label2 = label1.copy()
            label1[label1 != 1] = 0
            label2[label2 != 2] = 0
            # 归一化
            image = np.expand_dims(self.transform(image), axis=0)
            label1 = np.expand_dims(self.transform(label1), axis=0)
            label2 = np.expand_dims(self.transform(label2), axis=0)
            label = np.concatenate([label1, label2], axis=0)
            return torch.tensor(image).float(), torch.tensor(label).float()
        else:
            image = sitk.GetArrayFromImage(resize_image_itk(sitk.ReadImage(self.images[item]), (64, 64, 64)))
            image = np.expand_dims(self.transform(image), axis=0)
            return torch.tensor(image).float()


if __name__ == '__main__':

    h = Hippocampus(dirname='../datasets/3d', train=True)
    batch_size = 1
    dataloader = DataLoader(h, batch_size=batch_size, shuffle=False, num_workers=0)
    for i, img in enumerate(dataloader):
        print(i)
