from models.base_model import BaseModel


class CycleGANModel(BaseModel):
    """
    Implementation of CycleGAN model for image-to-image translation of unpaired
    data.

    
    """
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

    def set_input(self):
        pass

    def build(self):
        pass

    def __loss(self):
        pass

    def __optimizer(self):
        pass

    def load(self):
        pass

    def save(self):
        pass
