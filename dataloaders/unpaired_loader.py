import random
from PIL import Image
from dataloaders import BaseLoader
from utils import im_utils, file_utils


class UnpairedLoader(BaseLoader):
    """
    Dataloader meant for loading in two unpaired sets of images.

    Image set A is loaded in from path set by argument '--dirA /path/to/dataA'
    Image set B is loaded in from path set by argument '--dirB /path/to/dataB'
    """
    def __init__(self, opt):
        BaseLoader.__init__(self, opt)
        self.dirA = opt.dirA
        self.dirB = opt.dirB
        self.pathsA = file_utils.load_paths(self.dirA)
        self.pathsB = file_utils.load_paths(self.dirB)
        self.indexA = 0
        self.indexB = 0

    def __len__(self):
        return max(len(self.pathsA), len(self.pathsB))

    def __iter__(self):
        return self

    def __next__(self):
        pathA = self.pathsA[self.indexA]
        pathB = self.pathsB[self.indexB]

        imgA = Image.open(pathA)
        imgA = im_utils.augment(imgA, self.opt, grayscale=(self.opt.in_channels==1))
        imgB = Image.open(pathB)
        imgB = im_utils.augment(imgB, self.opt, grayscale=(self.opt.out_channels==1))

        self.indexA += 1
        self.indexB += 1

        if self.indexA == len(self.pathsA):
            self.indexA = 0
            random.shuffle(self.pathsA)

        if self.indexB == len(self.pathsB):
            self.indexB = 0
            random.shuffle(self.pathsB)

        return imgA, imgB
