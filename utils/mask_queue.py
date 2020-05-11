import random

"""
Code modified from original provided in xw-hu implementation of Mask-ShadowGAN.
https://github.com/xw-hu/Mask-ShadowGAN/blob/master/utils.py
"""

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
