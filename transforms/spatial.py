import random  
import torch.nn as nn  
from torchvision.transforms import v2

class processing(object):
    def __init__(self, size, is_aug=True, center=False):
        self.is_aug = is_aug
        self.center = center
        self.size = size
        self.f1_1 = v2.CenterCrop(size=(size, size))
        self.f1_2 = v2.RandomCrop(size=(size, size))
        self.f2 = v2.RandomApply(
            nn.ModuleList([
            v2.RandomChoice([
                v2.RandomHorizontalFlip(p=1),
                v2.RandomRotation(degrees=20),
                v2.ColorJitter(brightness=0.5)
            ])
            ]),
            p=0.3
        )
        self.state = random.getstate()

    def __call__(self, img):
        img = img/255.0
        c, h, w = img.shape
        self.state = random.getstate()

        pad_h = v2.Resize(size=(self.size, w), antialias=True)
        pad_w = v2.Resize(size=(h, self.size), antialias=True)
        if h < self.size:
            img = pad_h(img)
        if w < self.size:
            img = pad_w(img)
        if self.center:
            img = self.f1_1(img)
        else:
            img = self.f1_2(img)
        if self.is_aug:
            img = self.f2(img)

        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
        img = v2.Normalize(mean=mean, std=std)(img)
        random.setstate(self.state)
        return img
