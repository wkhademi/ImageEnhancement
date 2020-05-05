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
    def build(self):
        raise NotImplementedError
