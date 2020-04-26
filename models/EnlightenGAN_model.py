from models.base_model import BaseModel


class EnlightenGANModel(BaseModel):
    """
    Implementation of EnlightenGAN model for low-light image enhancement with
    unpaired data.


    """
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

    def build(self):
        pass
