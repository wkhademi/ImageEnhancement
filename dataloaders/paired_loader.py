from PIL import Image
from dataloaders import BaseLoader
from utils import im_utils, file_utils


class PairedLoader(BaseLoader):
    def __init__(self, opt):
        BaseLoader.__init__(self, opt)

    def __len__(self):
        return len(self.paths)

    def __iter__(self):
        return self

    def __next__(self):
        pass
