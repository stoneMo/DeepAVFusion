import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode
import numpy as np

class Compose(T.Compose):
    def __call__(self, img, lbls=()):
        for t in self.transforms:
            img, lbls = t(img, lbls)
        return img, lbls



class RandomResizedCrop(T.RandomResizedCrop):
    def forward(self, img, lbls=()):
        if not isinstance(lbls, (list, tuple)):
            lbls = [lbls]

        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias)
        lbls = [F.resized_crop(lbl, i, j, h, w, self.size, InterpolationMode.NEAREST, antialias=self.antialias) for lbl in lbls]
        return img, lbls


class Resize(T.Resize):
    def forward(self, img, lbls=()):
        if not isinstance(lbls, (list, tuple)):
            lbls = [lbls]
        img = F.resize(img, self.size, self.interpolation, self.max_size, self.antialias)
        lbls = [F.resize(lbl, self.size, InterpolationMode.NEAREST, self.max_size, self.antialias) for lbl in lbls]
        return img, lbls


class CenterCrop(T.CenterCrop):
    def forward(self, img, lbls=()):
        if not isinstance(lbls, (list, tuple)):
            lbls = [lbls]
        img = F.center_crop(img, self.size)
        lbls = [F.center_crop(lbl, self.size) for lbl in lbls]
        return img, lbls


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, img, lbls=()):
        if not isinstance(lbls, (list, tuple)):
            lbls = [lbls]
        if torch.rand(1) < self.p:
            return F.hflip(img), [F.hflip(lbl) for lbl in lbls]
        return img, lbls


class ToTensor(T.ToTensor):
    def __call__(self, img, lbls=()):
        return F.to_tensor(img), [torch.tensor(np.array(lbl)).long() for lbl in lbls]


class Normalize(T.Normalize):
    def forward(self, img, lbls=()):
        return F.normalize(img, self.mean, self.std, self.inplace), lbls

