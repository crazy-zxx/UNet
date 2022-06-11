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

        self.image_size = (512, 512, 128)
        self.patch_size = (64, 64, 64)
        self.step = (32, 32, 32)

        self.train = train
        if self.train:
            self.path = f'{dirname}/train'
        else:
            self.path = f'{dirname}/test'
            self.step = self.patch_size
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
        return np.prod((np.array(self.image_size) - np.array(self.patch_size)) / self.step + 1).astype(int) \
               * len(self.images)

    def normalization(self, data):
        a = data - np.min(data)
        b = np.max(data) - np.min(data)
        return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b != 0)

    def crop(self, data):
        # Tensor.unfold(dimension, size, step)
        windows_unpacked = data.unfold(1, self.patch_size[0], self.step[0]) \
            .unfold(2, self.patch_size[1], self.step[1]) \
            .unfold(3, self.patch_size[2], self.step[2])
        # crop
        windows = windows_unpacked.permute(1, 2, 3, 0, 4, 5, 6). \
            reshape(-1, self.patch_size[0], self.patch_size[1], self.patch_size[2])

        return windows

    def __getitem__(self, item):
        # which image
        per_image_patchs = self.__len__() / len(self.images)
        i = int(item / per_image_patchs + 1)
        j = int(item - (i - 1) * per_image_patchs)
        if self.train:
            # 图像resize
            image = sitk.GetArrayFromImage(resize_image_itk(sitk.ReadImage(self.images[i-1]), self.image_size))
            label = sitk.GetArrayFromImage(resize_image_itk(sitk.ReadImage(self.labels[i-1]), self.image_size))
            # 归一化
            image = torch.tensor(self.normalization(image)).unsqueeze(dim=0).float()
            label = torch.tensor(label).unsqueeze(dim=0).float()
            # crop
            image = self.crop(image).numpy()
            label = self.crop(label).numpy()
            image_crop = image[j]
            label_crop = label[j]
            # label图像分割多通道
            label_crop = mask2onehot(label_crop, classes_label)

            return torch.tensor(image_crop).unsqueeze(dim=0).float(), torch.tensor(label_crop).float()
        else:
            # 图像resize
            image = sitk.GetArrayFromImage(resize_image_itk(sitk.ReadImage(self.images[i-1]), self.image_size))
            # 归一化
            image = torch.tensor(self.normalization(image)).unsqueeze(dim=0).float()
            # crop
            image = self.crop(image).numpy()
            image_crop = image[j]

            return torch.tensor(image_crop).unsqueeze(dim=0).float()


if __name__ == '__main__':

    h = Abdomen(dirname=r'E:\datasets\Abdomen', train=True)
    batch_size = 2
    dataloader = DataLoader(h, batch_size=batch_size, shuffle=False, num_workers=0)
    for image, label in dataloader:
        print(image.shape, label.shape)
        break
