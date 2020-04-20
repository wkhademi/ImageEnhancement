from abc import ABC, abstractmethod


class BaseTest(ABC):
    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def test(self):
        pass
