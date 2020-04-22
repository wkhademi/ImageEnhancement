from abc import ABC, abstractmethod


class BaseDataset(object):
    """
    Abstract Base Class for defining the necessary methods needed for a
    TensorFlow dataset.
    """
    def __init__(self, opt, training):
        self.opt = opt
        self.training = training

    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def _preprocess(self):
        pass
