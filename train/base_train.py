from abc import ABC, abstractmethod


class BaseTrain(ABC):
    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def train(self):
        pass
