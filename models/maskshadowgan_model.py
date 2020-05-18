import tensorflow as tf
from models.base_model import BaseModel
from utils.im_utils import batch_convert_2_int
from datasets.single_dataset import SingleDataset
from datasets.unpaired_dataset import UnpairedDataset
from models.generators.maskshadowgan_generators import Generator
from models.discriminators.maskshadowgan_discriminators import Discriminator


class MaskShadowGANModel(BaseModel):
    """
    Implementation of Mask-ShadowGAN model for shadow removal of unpaired data.

    A: shadow images domain
    B: shadow free images domain

    Paper: https://arxiv.org/pdf/1903.10683.pdf
    """
    def __init__(self, opt, training):
        BaseModel.__init__(self, opt, training)

        # create placeholders for images and shadow masks
        self.realA = tf.placeholder(tf.float32, shape=[self.opt.batch_size, self.opt.crop_size, self.opt.crop_size, self.opt.channels])
        self.realB = tf.placeholder(tf.float32, shape=[self.opt.batch_size, self.opt.crop_size, self.opt.crop_size, self.opt.channels])
        self.fakeA = tf.placeholder(tf.float32, shape=[self.opt.batch_size, self.opt.crop_size, self.opt.crop_size, self.opt.channels])
        self.fakeB = tf.placeholder(tf.float32, shape=[self.opt.batch_size, self.opt.crop_size, self.opt.crop_size, self.opt.channels])
        self.rand_mask = tf.placeholder(tf.float32, shape=[self.opt.batch_size, self.opt.crop_size, self.opt.crop_size, 1])
        self.last_mask = tf.placeholder(tf.float32, shape=[self.opt.batch_size, self.opt.crop_size, self.opt.crop_size, 1])
        self.mask_non_shadow = tf.constant(-1.0, shape=[self.opt.batch_size, self.opt.crop_size, self.opt.crop_size, 1])

    def generate_dataset(self):
        """
        Add ops for dataset loaders to graph
        """
        if self.training:
            dataset = UnpairedDataset(self.opt, self.training)
            datasetA, datasetB = dataset.generate(cacheA='./dataA.tfcache', cacheB='./dataB.tfcache')
            dataA_iter = datasetA.make_initializable_iterator()
            dataB_iter = datasetB.make_initializable_iterator()

            return dataA_iter, dataB_iter, dataA_iter.get_next(), dataB_iter.get_next()
        else:  # only need shadow dataset for testing
            dataset = SingleDataset(self.opt, self.training)
            datasetA = dataset.generate()
            dataA_iter = datasetA.make_initializable_iterator()

            return dataA_iter, dataA_iter.get_next()

    def build(self):
        """
        Build TensorFlow graph for MaskShadowGAN model.
        """
        # add ops for generator (A->B) to graph
        self.G = Generator(channels=self.opt.channels, ngf=self.opt.ngf, norm_type=self.opt.layer_norm_type,
                           init_type=self.opt.weight_init_type, init_gain=self.opt.weight_init_gain,
                           training=self.training, name='G')

        if self.training:
            # add ops for other generator (B->A) and discriminators to graph
            self.F = Generator(channels=self.opt.channels, ngf=self.opt.ngf,
                               norm_type=self.opt.layer_norm_type, init_type=self.opt.weight_init_type,
                               init_gain=self.opt.weight_init_gain, training=self.training, name='F')
            self.D_A = Discriminator(channels=self.opt.channels, ndf=self.opt.ndf,
                                     norm_type=self.opt.layer_norm_type, init_type=self.opt.weight_init_type,
                                     init_gain=self.opt.weight_init_gain, training=self.training, name='D_A')
            self.D_B = Discriminator(channels=self.opt.channels, ndf=self.opt.ndf,
                                     norm_type=self.opt.layer_norm_type, init_type=self.opt.weight_init_type,
                                     init_gain=self.opt.weight_init_gain, training=self.training, name='D_B')

            # generate fake images
            fakeB = self.G(self.realA)
            fakeA = self.F(self.realB, self.rand_mask)

            # generate reconstructed images
            reconstructedA = self.F(fakeB, self.last_mask)
            reconstructedB = self.G(fakeA)

            # generate identity mapping images
            identA = self.G(self.realB)
            identB = self.F(self.realA, self.mask_non_shadow)

            tf.summary.image('A/original', batch_convert_2_int(self.realA))
            tf.summary.image('B/original', batch_convert_2_int(self.realB))
            tf.summary.image('A/generated', batch_convert_2_int(fakeA))
            tf.summary.image('B/generated', batch_convert_2_int(fakeB))
            tf.summary.image('A/reconstructed', batch_convert_2_int(reconstructedA))
            tf.summary.image('B/reconstructed', batch_convert_2_int(reconstructedB))

            # add loss ops to graph
            Gen_loss, D_A_loss, D_B_loss = self.__loss(fakeA, fakeB, reconstructedA,
                                                       reconstructedB, identA, identB)

            # add optimizer ops to graph
            optimizers = self.__optimizers(Gen_loss, D_A_loss, D_B_loss)

            return fakeA, fakeB, optimizers, Gen_loss, D_A_loss, D_B_loss
        else:  # only need generator from A->B during testing
            fakeB = self.G(self.realA)
            return fakeB

    def __loss(self, fakeA, fakeB, reconstructedA, reconstructedB, identA, identB):
        """
        Compute the losses for the generators and discriminators.
        """
        # compute the generators loss
        G_loss = self.__G_loss(self.D_B, fakeB)
        F_loss = self.__G_loss(self.D_A, fakeA)
        cc_loss = self.__cycle_consistency_loss(reconstructedA, reconstructedB)
        ident_loss = self.__identity_loss(identA, identB)
        Gen_loss = G_loss + F_loss + cc_loss + ident_loss

        # Compute the disciminators loss. Use fake images from image pool to improve stability
        D_A_loss = self.__D_loss(self.D_A, self.realA, self.fakeA)
        D_B_loss = self.__D_loss(self.D_B, self.realB, self.fakeB)

        return Gen_loss, D_A_loss, D_B_loss

    def __D_loss(self, D, real, fake):
        """
        Compute the discriminator loss.

        (MSE Loss):
            L_disc = 0.5 * [Expectation of (D(B) - 1)^2 + Expectation of (D(G(A)))^2]
        """
        loss = 0.5 * (tf.reduce_mean(tf.squared_difference(D(real), 1.0)) + \
                      tf.reduce_mean(tf.square(D(fake))))

        return loss

    def __G_loss(self, D, fake):
        """
        Compute the generator loss.

        (MSE Loss):
            L_gen = Expectation of (D(G(A)) - 1)^2
        """
        loss = tf.reduce_mean(tf.squared_difference(D(fake), 1.0))

        return loss

    def __cycle_consistency_loss(self, reconstructedA, reconstructedB):
        """
        Compute the cycle consistenty loss.

        L_cyc = lamA * [Expectation of L1_norm(F(G(A)) - A)] +
                lamb * [Expectation of L1_norm(G(F(B)) - B)]
        """
        loss = self.opt.lamA * tf.reduce_mean(tf.abs(reconstructedA - self.realA)) + \
               self.opt.lamB * tf.reduce_mean(tf.abs(reconstructedB - self.realB))

        return loss

    def __identity_loss(self, identA, identB):
        """
        Compute the identity loss.

        L_idt = lamda_idt * [lamA * [Expectation of L1_norm(F(A) - A)] +
                             lamB * [Expectation of L1_norm(G(B) - B)]]
        """
        loss = self.opt.lambda_ident * (self.opt.lamA * tf.reduce_mean(tf.abs(identB - self.realA)) + \
                                        self.opt.lamB * tf.reduce_mean(tf.abs(identA - self.realB)))

        return loss

    def __optimizers(self, Gen_loss, D_A_loss, D_B_loss):
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

        Gen_optimizer = make_optimizer(Gen_loss, self.G.variables + self.F.variables, name='Adam_Gen')
        D_A_optimizer = make_optimizer(D_A_loss, self.D_A.variables, name='Adam_D_A')
        D_B_optimizer = make_optimizer(D_B_loss, self.D_B.variables, name='Adam_D_B')

        with tf.control_dependencies([Gen_optimizer, D_A_optimizer, D_B_optimizer]):
            return tf.no_op(name='optimizers')
