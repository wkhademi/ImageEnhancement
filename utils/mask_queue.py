import random
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu

"""
Code modified from original provided in xw-hu implementation of Mask-ShadowGAN.
https://github.com/xw-hu/Mask-ShadowGAN/blob/master/utils.py
"""


def mask_generator(shadow, shadow_free):
    im_f = Image.fromarray(((np.squeeze(shadow_free, axis=0) + 1.0) * 127.5).astype(np.uint8)).convert('L')
    im_s = Image.fromarray(((np.squeeze(shadow, axis=0) + 1.0) * 127.5).astype(np.uint8)).convert('L')

    # difference between shadow image and shadow free image
    diff = np.asarray(im_f, dtype=np.float32) - np.asarray(im_s, dtype=np.float32)
    L = threshold_otsu(diff)
    mask = (np.float32(diff >= L) - 0.5) / 0.5

    return mask[None,:,:,None]


class MaskQueue:
    def __init__(self, queue_size):
        self.queue_size = queue_size
        self.queue = []

    def insert(self, mask):
        if len(self.queue) >= self.queue_size:
            self.queue.pop(0)

        self.queue.append(mask)

    def rand_item(self):
        idx = random.randint(0, len(self.queue)-1)
        return self.queue[idx]

    def last_item(self):
        return self.queue[-1]
