import tensorflow as tf
from models.base_model import BaseModel
from utils.im_utils import batch_convert_2_int
from datasets.single_dataset import SingleDataset
from datasets.unpaired_dataset import UnpairedDataset
from models.generators.enlightengan_generators import Generator
from models.discriminators.enlightengan_discriminators import Discriminator


class EnlightenGANModel(BaseModel):
    """
    Implementation of EnlightenGAN model for low-light image enhancement with
    unpaired data.

    Paper: https://arxiv.org/pdf/1906.06972.pdf
    """
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # # create dataset loaders
        # if training:
        #     self.dataset = UnpairedDataset(opt, training)
        #     self.datasetA, self.datasetB = self.dataset.generate(cacheA='./dataA.tfcache', cacheB='./dataB.tfcache')
        #     self.dataA_iter = self.datasetA.make_initializable_iterator()
        #     self.dataB_iter = self.datasetB.make_initializable_iterator()
        #     self.low = self.dataA_iter.get_next()
        #     self.normal = self.dataB_iter.get_next()
        # else:
        #     self.dataset = SingleDataset(opt, training)
        #     self.datasetA = self.dataset.generate()
        #     self.dataA_iter = self.datasetA.make_initializable_iterator()
        #     self.low = self.dataA_iter.get_next()

    def build(self):
        #normal = self.low  # NOTE: placeholder should be output of generator

        # build feature loss
        self.vgg16 = tf.keras.applications.VGG16(include_top=False,
                                                 input_shape=(self.opt.crop_size, self.opt.crop_size, 3))
        self.vgg16.trainable = False
        self.vgg16_features = self.vgg16.get_layer(self.opt.vgg_choose).output
        print(self.vgg16.get_layer(self.opt.vgg_choose).get_weights())

    def __loss():
        pass

    def __perceptual_loss(low, normal):
        """
        Compute the self feature preserving loss on the low-light and enhanced image.

        L_sfp =
        """
        pass

    def __vgg16_features(low, normal):
        """
        Extract features from VGG16 model.
        """
        vgg16_low_in = tf.keras.applications.vgg16.preprocess_input((low+1)*127.5)
        vgg16_normal_in = tf.keras.applications.vgg16.preprocess_input((normal+1)*127.5)

        vgg16_features_low = self.vgg16_features(vgg16_low_in)
        vgg16_features_normal = self.vgg16_features(vgg16_normal_out)

        return vgg16_features_low, vgg16_features_normal
