import sys
import random
from PIL import Image
from utils import im_utils, file_utils
from dataloaders.base_loader import BaseLoader


class PairedLoader(BaseLoader):
    """
        Data loader meant for loading in two paired sets of images. Assumes that
        image pairs have the same name (i.e. /path/to/dataA/10.png and /path/to/dataB/10.png).

        Image set A is loaded in from path set by argument '--dirA /path/to/dataA'
        Image set B is loaded in from path set by argument '--dirB /path/to/dataB'
    """
    def __init__(self, opt):
        BaseLoader.__init__(self, opt)
        self.dirA = opt.dirA
        self.dirB = opt.dirB
        self.pathsA = file_utils.load_paths(self.dirA)
        self.pathsB = file_utils.load_paths(self.dirB)

        if len(self.pathsA) != len(self.pathsB):
            print('Paired loader requires two datasets of equal length.')
            sys.exit(0)
        else:
            self.paths = zip(self.pathsA, self.pathsB)

        self.index = 0

    def __len__(self):
        return len(self.paths)

    def __iter__(self):
        return self

    def __next__(self):
        pathA, pathB = self.paths[self.index]

        imgA = Image.open(pathA)
        imgA = im_utils.augment(imgA, self.opt, grayscale=(self.opt.channels==1))
        imgB = Image.open(pathB)
        imgB = im_utils.augment(imgB, self.opt, grayscale=(self.opt.channels==1))

        self.index += 1

        if self.index == len(self.paths):
            self.index = 0
            random.shuffle(self.paths)

        return imgA, imgB
