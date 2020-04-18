import random
from PIL import Image
from dataloaders import BaseLoader
from utils import im_utils, file_utils


class UnpairedLoader(BaseLoader):
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

        imgA = Image.open(pathA).convert('RGB')
        imgA = im_utils.augment(imgA, self.opt)
        imgB = Image.open(pathB).convert('RGB')
        imgB = im_utils.augment(imgB, self.opt)

        self.indexA += 1
        self.indexB += 1

        if self.indexA == len(self.pathsA):
            self.indexA = 0
            random.shuffle(self.pathsA)

        if self.indexB == len(self.pathsB):
            self.indexB = 0
            random.shuffle(self.pathsB)

        return imgA, imgB
