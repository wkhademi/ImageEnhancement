import tensorflow as tf
from models.base_model import BaseModel
from utils.im_utils import batch_convert_2_int
from datasets.single_dataset import SingleDataset
from datasets.unpaired_dataset import UnpairedDataset
from utils.ops import __instance_normalization as instance_norm
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

        # create dataset loaders
        if training:
            self.dataset = UnpairedDataset(opt, training)
            self.datasetA, self.datasetB = self.dataset.generate(cacheA='./dataA.tfcache', cacheB='./dataB.tfcache')
            self.dataA_iter = self.datasetA.make_initializable_iterator()
            self.dataB_iter = self.datasetB.make_initializable_iterator()
            self.low = self.dataA_iter.get_next()
            self.normal = self.dataB_iter.get_next()
        else:
            self.dataset = SingleDataset(opt, training)
            self.datasetA = self.dataset.generate()
            self.dataA_iter = self.datasetA.make_initializable_iterator()
            self.low = self.dataA_iter.get_next()

    def build(self):
        # add ops for Generator (low light -> normal light) to graph
        self.G = Generator()

        if self.training:
            # build feature extractor if necessary
            if self.opt.vgg:
                self.vgg16 = tf.keras.applications.VGG16(include_top=False,
                                                         input_shape=(self.opt.crop_size, self.opt.crop_size, 3))
                self.vgg16.trainable = False
                self.vgg16_features = self.vgg16.get_layer(self.opt.vgg_choose).output

            # build patch feature extractor if necessary
            if self.opt.patch_vgg:
                self.vgg16_patch = tf.keras.applications.VGG16(include_top=False,
                                                               input_shape=(self.opt.patch_size, self.opt.patch_size, 3))
                self.vgg16_patch.trainable = False
                self.vgg16_patch_features = self.vgg16_patch.get_layer(self.opt.vgg_choose).output

            # add loss ops to graph
            _ = self.__loss()

            # add optimizer ops to graph
            _ = self.__optimizers()

            return None
        else:
            enhanced = self.G(self.low)
            return enhanced

    def __loss():
        pass

    def __perceptual_loss(low, enhanced, low_patch, enhanced_patch, low_patches, enhanced_patches):
        """
        Compute the self feature preserving loss on the low-light and enhanced image.
        """
        loss_feature_patch = 0
        features_low = self.__vgg16_features(low)
        features_normal = self._vgg16_features(enhanced)

        if self.opt.patch_vgg:
            features_low_patch = self.__vgg16_patch_features(low_patch)
            features_normal_patch = self.__vgg16_patch_features(enhanced_patch)

        if self.opt.patchD_3:
            features_low_patches = self.__vgg16_patch_features(low_patches)
            features_normal_patches = self.__vgg16_patch_features(enhanced_patches)

        if self.opt.no_vgg_instance:
            loss = tf.reduce_mean(tf.squared_difference(features_low, features_normal))

            if self.opt.patch_vgg:
                loss_feature_patch += tf.reduce_mean(tf.squared_difference(features_low_patch,
                                                                           features_normal_patch))

            if self.opt.patchD_3:
                loss_feature_patch += tf.reduce_mean(tf.squared_difference(features_low_patches,
                                                                           features_normal_patches))
        else:
            loss = tf.reduce_mean(tf.squared_difference(instance_norm(features_low),
                                                        instance_norm(features_normal)))

            if self.opt.patch_vgg:
                loss_feature_patch += tf.reduce_mean(tf.squared_difference(instance_norm(features_low_patch),
                                                                           instance_norm(features_normal_patch)))

            if self.opt.patchD_3:
                loss_feature_patch += tf.reduce_mean(tf.squared_difference(instance_norm(features_low_patches),
                                                                           instance_norm(features_normal_patches)))

        loss += loss_feature_patch / (self.opt.patchD_3 + 1)

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

    def __vgg16_features(image):
        """
        Extract features from image using VGG16 model.
        """
        vgg16_in = tf.keras.applications.vgg16.preprocess_input((image+1)*127.5)
        vgg16_features = self.vgg16_features(vgg16_in)

        return vgg16_features

    def __vgg16_patch_features(image):
        """
        Extract features from patches using VGG16 model.
        """
        vgg16_in = tf.keras.applications.vgg16.preprocess_input((image+1)*127.5)
        vgg16_features = self.vgg16_patch_features(vgg16_in)

        return vgg16_features
