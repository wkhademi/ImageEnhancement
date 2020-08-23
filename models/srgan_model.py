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
        pass

    def __optimizers(self, Gen_loss, D_loss, D_P_loss=None):
        """
        Modified optimizer taken from vanhuyz TensorFlow implementation of CycleGAN
        https://github.com/vanhuyz/CycleGAN-TensorFlow/blob/master/model.py
        """
        def make_optimizer(loss, variables, name='Adam'):
            """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
                and a linearly decaying rate that goes to zero over the next 100k steps
            """
            global_step = tf.Variable(0, trainable=False, name='global_step')
            starter_learning_rate = self.opt.lr
            end_learning_rate = 0.0
            start_decay_step = self.opt.niter
            decay_steps = self.opt.niter_decay
            beta1 = self.opt.beta1
            learning_rate = (tf.where(tf.greater_equal(global_step, start_decay_step),
                                      tf.train.polynomial_decay(starter_learning_rate,
                                                                global_step-start_decay_step,
                                                                decay_steps, end_learning_rate,
                                                                power=1.0),
                                      starter_learning_rate))

            learning_step = (tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
                                .minimize(loss, global_step=global_step, var_list=variables))

            return learning_step

        Gen_optimizer = make_optimizer(Gen_loss, self.G.variables, name='Adam_Gen')
        D_optimizer = make_optimizer(D_loss, self.D.variables, name='Adam_D')

        optimizers = [Gen_optimizer, D_optimizer]

        if D_P_loss is not None:
            D_P_optimizer = make_optimizer(D_P_loss, self.D_P.variables, name='Adam_D_P')
            optimizers.append(D_P_optimizer)

        with tf.control_dependencies(optimizers):
            return tf.no_op(name='optimizers')
