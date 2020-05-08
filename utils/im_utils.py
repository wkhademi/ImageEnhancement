import sys
import random
import numpy as np
import tensorflow as tf
from PIL import Image


def augment(img, opt, grayscale=False):
    img = convert_2_grayscale(img) if grayscale else convert_2_rgb(img)

    if self.opt.scale == 'power_2':
        img = make_power_2(img, 4)
    elif self.opt.scale == 'center_crop':
        img = center_crop(img, self.opt.crop_size, self.opt.crop_size)
    elif self.opt.scale == 'random_crop':
        img = random_crop(img, self.opt.crop_size, self.opt.crop_size)

    if self.opt.flip:
        orientation = random.randint(0,2)
        img = flip(img, flip=orientation)

    img = np.array(img, dtype=np.float32)[None,:,:,:]

    if self.norm_type == 'subtract_mean':
        img = subtract_mean(img, mean=self.opt.norm_mean, per_channel=self.opt.per_channel_norm)
    elif self.norm_type == 'normalize':
        img = normalize(img, min_val=self.opt.norm_min, max_val=self.opt.norm_max)
    elif self.norm_type == 'standardize':
        img = standardize(img, mean=self.opt.norm_mean, per_channel=self.opt.per_channel_norm,
                            min_val=self.opt.norm_min, max_val=self.opt.norm_max)

    return img


def convert_2_grayscale(img):
    return img.convert('L')


def convert_2_rgb(img):
    return img.convert('RGB')


def convert_2_tfint(img):
    img = (img + 1.) / 2.
    return tf.image.convert_image_dtype(img, dtype=tf.uint8)


def batch_convert_2_int(imgs):
    return tf.map_fn(convert_2_tfint, imgs, dtype=tf.uint8)


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

    left = random.randint(0, int(width - crop_width // 2) - 1)
    upper = random.randint(0, int(height - crop_height // 2) - 1)
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
