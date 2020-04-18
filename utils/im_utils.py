import sys
import random
import numpy as np
from PIL import Image


def augment(img, opt):
    pass


def make_power_2(img, base, method=Image.BICUBIC):
    width, height = img.size

    new_width = int(width // base) * base
    new_height = int(height // base) * base

    if width == new_width and height == new_height:
        return img

    return img.resize((new_width, new_height), resample=method)


def center_crop(img, crop_height, crop_width):
    width, height = img.size

    if crop_width > width or crop_height > height:
        print("Requested crop greater than image size")
        sys.exit(0)

    left = int(width - crop_width // 2)
    upper = int(height - crop_height // 2)
    right = left + crop_width
    lower = upper + crop_height

    return img.crop((left, upper, right, lower))


def random_crop(img, crop_height, crop_width):
    width, height = img.size

    if crop_width > width or crop_height > height:
        print("Requested crop greater than image size")
        sys.exit(0)

    left = random.randint(0, int(width - crop_width // 2))
    upper = random.randint(0, int(height - crop_height // 2))
    right = left + crop_width
    lower = upper + crop_height

    return img.crop((left, upper, right, lower))


def flip(img, flip=0):
    if flip == 0:
        return img
    elif flip == 1:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip == 2:
        return img.transpose(Image.FLIP_TOP_BOTTOM)


def subtract_mean(imgs, mean=None, per_channel=False):
    if mean is None:
        if per_channel:
            imgs = imgs - np.mean(imgs, axis=(1, 2))
        else:
            imgs = imgs - np.mean(imgs)
    else:
        imgs = imgs - mean

    return imgs


def normalize(imgs, min_val=0, max_val=255):
    imgs = (imgs - min_val) / (max_val - min_val)

    return imgs


def standardize(imgs, mean=None, per_channel=False, min_val=0, max_val=255):
    imgs = subtract_mean(imgs, mean=mean, per_channel=per_channel)
    imgs = normalize(imgs, min_val=min_val, max_val=max_val)

    return imgs
