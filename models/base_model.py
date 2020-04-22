from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract Base Class for defining the necessary methods needed for an
    Image Enhancement model.
    """
    def __init__(self, opt, training):
        self.opt = opt
        self.training = training

    @abstractmethod
    def set_input(self):
        raise NotImplementedError

    @abstractmethod
    def build(self):
        raie NotImplementedError

    @abstractmethod
    def __loss(self):
        raise NotImplementedError

    @abstractmethod
    def __optimizer(self):
        raise NotImplementedError

    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def save(self):
        raise NotImplementedError
