from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract Base Class for defining the necessary methods needed for an
    Image Enhancement model.
    """
    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def input(self):
        raise NotImplementedError

    @abstractmethod
    def build(self):
        raie NotImplementedError

    @abstractmethod
    def loss(self):
        raise NotImplementedError

    @abstractmethod
    def optimizer(self):
        raise NotImplementedError
