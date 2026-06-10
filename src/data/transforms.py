import random
from PIL import Image, ImageFilter
import torch
from torchvision import transforms

class RandomGaussianBlur:
    """Aplicacion de Gaussian Blur con probabilidad p."""
    def __init__(self, p=0.5, radius_range=(0.1, 2.0)):
        self.p = p
        self.radius_range = radius_range

    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(*self.radius_range)
            # PIL implementation of Gaussian Blur
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img

class RandomUnsharpMask:
    """
    Aplicacion Unsharp Mask (sharpening/high-pass filter effect) con probabilidad p.
    """
    def __init__(self, p=0.5, radius_range=(0.5, 2.0), percent_range=(100, 200), threshold=3):
        self.p = p
        self.radius_range = radius_range
        self.percent_range = percent_range
        self.threshold = threshold

    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(*self.radius_range)
            percent = random.randint(*self.percent_range)
            return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=self.threshold))
        return img
