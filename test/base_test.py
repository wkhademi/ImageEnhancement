from abc import ABC, abstractmethod


class BaseTest(ABC):
    """
    Base trainer for any model.
    """
    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def test(self):
        pass
