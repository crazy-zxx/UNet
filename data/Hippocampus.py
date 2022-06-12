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
        # 受制于GPU显存，如果图像尺寸太大，则必须进行crop。大于（128，128，128）就建议进行切分！
        # 切分小块的大小
        self.patch_size = (64, 64, 64)
        # 按照步长来crop
        self.step = (64, 64, 64)

        self.train = train
        if self.train:
            self.path = f'{dirname}/train'
        else:
            self.path = f'{dirname}/test'
            self.step = self.patch_size

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
        # 数据集大小（如果crop，则是crop后的总块数）
        return np.prod((np.array(self.image_size) - np.array(self.patch_size)) / self.step + 1).astype(int) \
               * len(self.images)

    def normalization(self, data):
        """   数据归一化  """
        a = data - np.min(data)
        b = np.max(data) - np.min(data)
        # 避免除0操作
        return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b != 0)

    def crop(self, data):
        """     切分数据，此处是切分一张图像，输入张量大小：（1,x,y,z）    """
        # Tensor.unfold(dimension, size, step)
        windows_unpacked = data.unfold(1, self.patch_size[0], self.step[0]) \
            .unfold(2, self.patch_size[1], self.step[1]) \
            .unfold(3, self.patch_size[2], self.step[2])
        # crop
        windows = windows_unpacked.permute(1, 2, 3, 0, 4, 5, 6). \
            reshape(-1, self.patch_size[0], self.patch_size[1], self.patch_size[2])

        return windows

    def __getitem__(self, item):
        # 计算每张图切分的patch数目
        per_image_patchs = int(self.__len__() / len(self.images))
        # 计算所在图片的索引+1
        i = int(item / per_image_patchs + 1)
        # 计算所在的图像上的patch索引
        j = int(item - (i - 1) * per_image_patchs)
        if self.train:
            # 读取图像，并resize。数据shape会由image_size(x,y,z)变成(z,y,x) ！！！
            image = sitk.GetArrayFromImage(resize_image_itk(sitk.ReadImage(self.images[i - 1]), self.image_size))
            label = sitk.GetArrayFromImage(resize_image_itk(sitk.ReadImage(self.labels[i - 1]), self.image_size))
            # 图像归一化，标签不用归一化
            image = torch.tensor(self.normalization(image)).unsqueeze(dim=0).float()
            label = torch.tensor(label).unsqueeze(dim=0).float()
            # crop
            image = self.crop(image).numpy()
            label = self.crop(label).numpy()
            # 找到需要的patch
            image_crop = image[j]
            label_crop = label[j]
            # label分割多通道（onehot）
            label_crop = mask2onehot(label_crop, classes_label)
            # 返回patch
            return torch.tensor(image_crop).unsqueeze(dim=0).float(), torch.tensor(label_crop).float()
        else:
            # 读取图像，并resize
            image = sitk.GetArrayFromImage(resize_image_itk(sitk.ReadImage(self.images[i - 1]), self.image_size))
            # 图像归一化
            image = torch.tensor(self.normalization(image)).unsqueeze(dim=0).float()
            # crop
            image = self.crop(image).numpy()
            # 找到需要的patch
            image_crop = image[j]
            # 返回patch
            return torch.tensor(image_crop).unsqueeze(dim=0).float()


if __name__ == '__main__':

    h = Hippocampus(dirname='../datasets/3d/hippocampus', train=True)
    batch_size = 2
    dataloader = DataLoader(h, batch_size=batch_size, shuffle=False, num_workers=0)
    for image, label in dataloader:
        print(image.shape, label.shape)
        break
