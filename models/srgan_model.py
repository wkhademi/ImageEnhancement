import utils.ops as ops
import tensorflow as tf
from models.base_model import BaseModel
from utils.im_utils import batch_convert_2_int
from datasets.superres_dataset import SuperResDataset
from models.generators.srgan_generators import Generator
from models.discriminators.srgan_discriminators import Discriminator


class SRGANModel(BaseModel):
    """
    Implementation of SRGAN model for super-resolution.

    Paper: https://arxiv.org/pdf/1609.04802.pdf
    """
    def __init__(self, opt, training):
        BaseModel.__init__(self, opt, training)

        # create dataset loaders
        self.dataset = SuperResDataset(opt, training)
        self.data = self.dataset.generate(cache='./data.tfcache')
        self.data_iter = self.data.make_initializable_iterator()

        if training:
            self.highres, self.lowres = self.data_iter.get_next()
        else:
            self.lowres = self.data_iter.get_next()

    def build(self):
        """
        Build TensorFlow graph for SRGAN model.
        """
        # add ops for the generator to the graph
        self.G = Generator(channels=self.opt.channels, ngf=self.opt.ngf, norm_type=self.opt.layer_norm_type,
                           init_type=self.opt.weight_init_type, init_gain=self.opt.weight_init_gain,
                           dropout=self.opt.dropout, training=self.training, name='G')

        if self.training:
            # add ops for the discriminator to the graph
            self.D = Discriminator(channels=self.opt.channels, ndf=self.opt.ndf, norm_type=self.opt.layer_norm_type,
                                   init_type=self.opt.weight_init_type, init_gain=self.opt.weight_init_gain,
                                   training=self.training, name='D')

            # build feature extractor
            self.vgg19 = tf.keras.applications.VGG19(include_top=False, input_shape=(None, None, 3))
            self.vgg19.trainable = False

            superres = self.G(self.lowres)

            # add loss ops to graph
            Gen_init_loss, Gen_loss, D_loss = self.__loss(superres, self.highres)

            # add optimizer ops to graph
            gen_optimizer, optimizers = self.__optimizers(Gen_init_loss, Gen_loss, D_loss)

            return gen_optimizer, optimizers, Gen_init_loss, Gen_loss, D_loss
        else:
            superres = self.G(self.lowres)
            return superres

    def __loss(self, superres, highres):
        """
        Compute losses for generator and discriminators.
        """
        Gen_init_loss = self.__mse_loss(superres, highres)

        Gen_loss = self.__mse_loss(superres, highres) + \
                   self.__G_loss(self.D, superres) + \
                   self.__perceptual_loss(superres, highres)

        D_loss = self.__D_loss(self.D, superres, highres)

        return Gen_init_loss, Gen_loss, D_loss

    def __D_loss(self, D, superres, highres, eps=1e-12):
        """
        Compute the discriminator loss.

        L_disc = -0.5 * [Expectation of log(D(B)) + Expectation of log(1 - D(G(A)))]
        """
        loss = -1 * (tf.reduce_mean(tf.log(D(highres) + eps)) + \
                     tf.reduce_mean(tf.log(1 - D(superres) + eps)))

        return loss

    def __G_loss(self, D, superres, eps=1e-12):
        """
        Compute the generator loss.

        L_gen = Expectation of -log(D(G(A)))
        """
        loss = -1e-3 * tf.reduce_mean(tf.log(D(superres) + eps))

        return loss

    def __mse_loss(self, superres, highres):
        """
        Compute the pixel-wise mean squared error.
        """
        loss = tf.reduce_mean(tf.square(superres - highres))

        return loss

    def __perceptual_loss(self, superres, highres):
        """
        Compute the perceptual loss on super-resolution and high-resolution images.
        """
        features_superres = self.__vgg19_features(superres)
        features_highres = self.__vgg19_features(highres)

        loss = 6e-3 * tf.reduce_mean(tf.square(features_superres - features_highres))

        return loss

    def __vgg19_features(self, image):
        """
        Extract features from image using VGG19 model.
        """
        vgg19_in = tf.keras.applications.vgg19.preprocess_input((image+1)*127.5)
        x = vgg19_in

        for i in range(len(self.vgg19.layers)):
            x = self.vgg19.layers[i](x)

            if self.vgg19.layers[i].name == self.opt.vgg_choose:
                break

        vgg19_features = x

        return vgg19_features

    def __optimizers(self, Gen_init_loss, Gen_loss, D_loss):
        """
        Create optimizers for generator and discriminator.
        """
        self.lr = tf.Variable(self.opt.lr, trainable=False)
        self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_lr")
        self.update_lr = tf.assign(self.lr, self.new_lr)

        Gen_optimizer = tf.train.AdamOptimizer(self.lr, beta1=self.opt.beta1, name='Adam_Gen')
                            .minimize(Gen_loss, var_list=self.G.variables)
        D_optimizer = tf.train.AdamOptimizer(self.lr, beta1=self.opt.beta1, name='Adam_D')
                            .minimize(D_loss, var_list=self.D.variables)

        optimizers = [Gen_optimizer, D_optimizer]

        with tf.control_dependencies(optimizers):
            return Gen_optimizer, tf.no_op(name='optimizers')
