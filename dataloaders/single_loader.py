import random
from PIL import Image
from utils import im_utils, file_utils
from dataloaders.base_loader import BaseLoader


class SingleLoader(BaseLoader):
    """
    Data loader meant for loading a single set of images.

    Images are loaded from the path set by argument '--dir /path/to/data'
    """
    def __init__(self, opt):
        BaseLoader.__init__(self, opt)
        self.dir = opt.dir
        self.paths = file_utils.load_paths(self.dir)
        self.index = 0

    def __len__(self):
        return len(self.paths)

    def __iter__(self):
        return self

    def __next__(self):
        path = self.paths[self.index]
        img = Image.open(path)
        img = im_utils.augment(img, self.opt, grayscale=(self.opt.in_channels==1))

        self.index += 1

        if self.index == self.__len__():
            self.index = 0
            random.shuffle(self.paths)

        return img
