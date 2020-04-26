from models.base_model import BaseModel


class EnlightenGANModel(BaseModel):
    """
    Implementation of DeshadowNet model for shadow removal.


    """
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

    def build(self):
        pass
