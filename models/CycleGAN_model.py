import tensorflow as tf
from models.base_model import BaseModel
from datasets.single_dataset import SingleDataset
from datasets.unpaired_dataset import UnpairedDataset


class CycleGANModel(BaseModel):
    """
    Implementation of CycleGAN model for image-to-image translation of unpaired
    data.
    """
    def __init__(self, opt, training):
        BaseModel.__init__(self, opt, training)

        # create dataset loaders
        if training:
            self.dataset = UnpairedDataset(opt, training)
            self.datasetA, self.datasetB = self.dataset.generate(cacheA='./dataA.tfcache', cacheB='./dataB.tfcache')
            self.dataA_iter = self.datasetA.make_initializable_iterator()
            self.dataB_iter = self.datasetB.make_initializable_iterator()
        else:
            self.dataset = SingleDataset(opt, training)
            self.datasetA = dataset.generate()
            self.dataA_iter = self.datasetA.make_initializable_iterator()

        # create placeholders for fake images
        self.fakeA = tf.placeholder(tf.float32,
            shape=[self.opt.batch_size, self.opt.crop_size, self.opt.crop_size, self.opt.in_channels])
        self.fakeB = tf.placeholder(tf.float32,
            shape=[self.opt.batch_size, self.opt.crop_size, self.opt.crop_size, self.opt.out_channels])

    def set_input(self):
        if self.training:  # use unpaired dataset loader when training
            self.realA = self.dataA_iter.get_next()
            self.realB = self.dataB_iter.get_next()
        else:  # use single dataset loader when testing
            self.realA = self.dataA_iter.get_next()

    def build(self):
        pass

    def __loss(self):
        pass

    def __optimizer(self):
        """
        Optimizer taken from vanhuyz TensorFlow implementation of CycleGAN
        https://github.com/vanhuyz/CycleGAN-TensorFlow/blob/master/model.py
        """
        def make_optimizer(loss, variables, name='Adam'):
            """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
                and a linearly decaying rate that goes to zero over the next 100k steps
            """
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.opt.learning_rate
            end_learning_rate = 0.0
            start_decay_step = 100000
            decay_steps = 100000
            beta1 = self.opt.beta1
            learning_rate = (
                tf.where(
                    tf.greater_equal(global_step, start_decay_step),
                    tf.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,
                                                decay_steps, end_learning_rate,
                                                power=1.0),
                    starter_learning_rate))

            learning_step = (
                tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
                    .minimize(loss, global_step=global_step, var_list=variables)
            )

            return learning_step

        G_optimizer = make_optimizer(G_loss, self.G.variables, name='Adam_G')
        D_B_optimizer = make_optimizer(D_B_loss, self.D_B.variables, name='Adam_D_B')
        F_optimizer =  make_optimizer(F_loss, self.F.variables, name='Adam_F')
        D_A_optimizer = make_optimizer(D_A_loss, self.D_A.variables, name='Adam_D_A')

        with tf.control_dependencies([G_optimizer, D_B_optimizer, F_optimizer, D_A_optimizer]):
            return tf.no_op(name='optimizers')

    def load(self):
        pass

    def save(self):
        pass
