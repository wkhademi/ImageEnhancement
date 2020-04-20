import argparse
from options.base_options import BaseOptions


def DeshadowNetOptions(BaseOptions):
    def __init__(self, training):
        BaseOptions.__init__(self)

    def parse(self):
        return self.parser.parse_args()
