import os

import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.oneHot import mask2onehot

classes_label = [0, 255]


class FruitFlyCell(Dataset):
    def __init__(self, dirname, train=True, transform=None):
        super(FruitFlyCell, self).__init__()
        self.transform = transform
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

    def __getitem__(self, item):
        if self.train:
            image = np.expand_dims(cv2.resize(cv2.imread(self.images[item], cv2.IMREAD_GRAYSCALE), [512, 512]), axis=2)
            label = mask2onehot(cv2.resize(cv2.imread(self.labels[item], cv2.IMREAD_GRAYSCALE), [512, 512]),
                                classes_label)
            if self.transform:
                image = self.transform(image)
            return image, label
        else:
            image = np.expand_dims(cv2.resize(cv2.imread(self.images[item], cv2.IMREAD_GRAYSCALE), [512, 512]), axis=2)
            if self.transform:
                image = self.transform(image)
            return image


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor()  # data-->[0,1]
    ])
    h = FruitFlyCell(dirname='../datasets/2d/cell', train=True, transform=transform)
    batch_size = 2
    dataloader = DataLoader(h, batch_size=batch_size, shuffle=False, num_workers=0)
    for img, label in dataloader:
        print(img.shape, label.shape)
        break
