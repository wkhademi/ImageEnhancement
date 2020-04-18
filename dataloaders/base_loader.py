class BaseLoader(object):
    def __init__(self, opt):
        self.opt = opt

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError
