from abc import ABC, abstractmethod


class BaseLoader(ABC):
    """
    Abstract Base Class for defining the necessary methods needed for a
    data loader.
    """
    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def __next__(self):
        raise NotImplementedError
