import random
from PIL import Image
from dataloaders import BaseLoader
from utils import im_utils, file_utils


class SingleLoader(BaseLoader):
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
        img = Image.open(path).convert('RGB')
        img = im_utils.augment(img, self.opt)

        self.index += 1

        if self.index == self.__len__():
            self.index = 0
            random.shuffle(self.paths)

        return img
