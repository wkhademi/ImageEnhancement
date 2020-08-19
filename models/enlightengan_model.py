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
        self.G = Generator(channels=self.opt.channels, netG=self.opt.netG, ngf=self.opt.ngf,
                           norm_type=self.opt.layer_norm_type, init_type=self.opt.weight_init_type,
                           init_gain=self.opt.weight_init_gain, dropout=self.opt.dropout,
                           self_attention=self.opt.self_attention, times_residual=self.opt.times_residual,
                           skip=self.opt.skip, training=self.training, name='G')

        if self.training:
            # add ops for discriminator to graph
            self.D = Discriminator(channels=self.opt.channels, netD=self.opt.netD, n_layers=self.opt.n_layers,
                                   ndf=self.opt.ndf, norm_type=self.opt.layer_norm_type,
                                   init_type=self.opt.weight_init_type, init_gain=self.opt.weight_init_gain,
                                   training=self.training, gan_mode=self.opt.gan_mode, name='D')

            # add ops for patch discriminator to graph if necessary
            if self.opt.patchD:
                self.D_P = Discriminator(channels=self.opt.channels, netD=self.opt.netD, n_layers=self.opt.n_layers_patch,
                                         ndf=self.opt.ndf, norm_type=self.opt.layer_norm_type,
                                         init_type=self.opt.weight_init_type, init_gain=self.opt.weight_init_gain,
                                         training=self.training, gan_mode=self.opt.gan_mode, name='D_P')

            # build feature extractor if necessary
            if self.opt.vgg:
                self.vgg16 = tf.keras.applications.VGG16(include_top=False,
                                                         input_shape=(None, None, 3))
                self.vgg16.trainable = False
                self.vgg16_features = self.vgg16.get_layer(self.opt.vgg_choose).output

            gray = 1. - tf.image.rgb_to_grayscale((self.low+1.)/2.)

            if self.opt.skip > 0:
                enhanced, latent_enhanced = self.G(self.low, gray)
            else:
                enhanced = self.G(self.low, gray)

            if self.opt.patchD:
                height = self.opt.crop_size
                width = self.opt.crop_size
                height_offset = tf.random.uniform(1, maxval=height-self.opt.patch_size-1, dtype=tf.int32)
                width_offset = tf.random.uniform(1, maxval=width-self.opt.patch_size-1, dtype=tf.int32)

                low_patch = ops.crop(self.low, height_offset, width_offset, self.opt.patch_size)
                normal_patch = ops.crop(self.normal, height_offset, width_offset, self.opt.patch_size)
                enhanced_patch = ops.crop(enhanced, height_offset, width_offset, self.opt.patch_size)
            else:
                low_patch = None
                normal_patch = None
                enhanced_patch = None

            if self.opt.patchD_3 > 0:
                height = self.opt.crop_size
                width = self.opt.crop_size
                height_offset = tf.random.uniform(self.opt.patchD_3, maxval=height-self.opt.patch_size-1, dtype=tf.int32)
                width_offset = tf.random.uniform(self.opt.patchD_3, maxval=width-self.opt.patch_size-1, dtype=tf.int32)

                height_offset = tf.tile(height_offset, [self.opt.batch_size])
                width_offset = tf.tile(width_offset, [self.opt.batch_size])
                low_patches = tf.repeat(self.low, repeats=self.opt.patchD_3, axis=0)
                normal_patches = tf.repeat(self.normal, repeats=self.opt.patchD_3, axis=0)
                enhanced_patches = tf.repeat(enhanced, repeats=self.opt.patchD_3, axis=0)

                low_patches = tf.map_fn(lambda x: ops.crop(x[0], x[1], x[2], self.opt.patch_size),
                                (low_patches, height_offset, width_offset), fn_output_signature=tf.float32)
                normal_patches = tf.map_fn(lambda x: ops.crop(x[0], x[1], x[2], self.opt.patch_size),
                                    (normal_patches, height_offset, width_offset), fn_output_signature=tf.float32)
                enhanced_patches = tf.map_fn(lambda x: ops.crop(x[0], x[1], x[2], self.opt.patch_size),
                                        (enhanced_patches, height_offset, width_offset), fn_output_signature=tf.float3)
            else:
                low_patches = None
                normal_patches = None
                enhanced_patches = None

            # add loss ops to graph
            Gen_loss, D_loss, D_P_loss = self.__loss(self.low, self.normal, enhanced, low_patch, normal_patch,
                                                     enhanced_patch, low_patches, normal_patches, enhanced_patches)

            # add optimizer ops to graph
            optimizers = self.__optimizers(Gen_loss, D_loss, D_P_loss)

            if D_P_loss is None:  # create dummy value to avoid error
                D_P_loss = tf.constant(0)

            return enhanced, optimizers, Gen_loss, D_loss, D_P_loss
        else:
            enhanced = self.G(self.low)[0] if self.opt.skip > 0 else self.G(self.low)
            return enhanced

    def __loss(low, normal, enhanced, low_patch, normal_patch, enhanced_patch,
               low_patches, normal_patches, enhanced_patches):
        """
        Compute losses for generator and discriminators.
        """
        use_ragan = False if self.opt.hybrid_loss else True  # for hybrid losss

        # compute generator loss
        Gen_loss = self.__G_loss(self.D, normal, enhanced, use_ragan=True)

        if self.opt.patchD:
            Gen_loss += self.__G_loss(self.D_P, normal_patch, enhanced_patch, use_ragan=use_ragan)

            if self.opt.patchD_3 > 0:
                Gen_loss += self.__G_loss(self.D_P, normal_patches, enhanced_patches, use_ragan=use_ragan)

        if self.opt.vgg:
            Gen_loss += self.__perceptual_loss(low, enhanced, low_patch, enhanced_patch,
                                               low_patches, enhanced_patches)

        # compute global discriminator loss
        D_loss = self.__D_loss(self.D, normal, enhanced, use_ragan=True)

        # compute local discriminator loss if necessary
        D_P_loss = None
        if self.opt.patchD:
            D_P_loss = self.__D_loss(self.D_P, normal_patch, enhanced_patch, use_ragan=use_ragan)

            if self.opt.patchD_3 > 0:
                D_P_loss += self.__D_loss(self.D_P, normal_patches, enhanced_patches, use_ragan=use_ragan)

        return Gen_loss, D_loss, D_P_loss

    def __D_loss(self, D, normal, enhanced, use_ragan=False, eps=1e-12):
        """
        Compute the discriminator loss.

        If LSGAN is used: (MSE Loss)
            L_disc = 0.5 * [Expectation of (D(B) - 1)^2 + Expectation of (D(G(A)))^2]
        Otherwise: (NLL Loss)
            L_disc = -0.5 * [Expectation of log(D(B)) + Expectation of log(1 - D(G(A)))]
        """
        if self.opt.use_ragan and use_ragan:
            loss = 0.5 * (tf.reduce_mean(tf.squared_difference(D(normal) - tf.reduce_mean(D(enhanced)), 1.0)) + \
                          tf.reduce_mean(tf.square(D(enhanced) - tf.reduce_mean(D(normal)))))
        elif self.opt.gan_mode == 'lsgan':
            loss = 0.5 * (tf.reduce_mean(tf.squared_difference(D(normal), 1.0)) + \
                          tf.reduce_mean(tf.square(D(enhanced))))
        elif self.opt.gan_mode == 'vanilla':
            loss = -0.5 * (tf.reduce_mean(tf.log(D(normal) + eps)) + \
                           tf.reduce_mean(tf.log(1 - D(enhanced) + eps)))

        return loss

    def __G_loss(self, D, normal, enhanced, use_ragan=False, eps=1e-12):
        """
        Compute the generator loss.

        If LSGAN is used: (MSE Loss)
            L_gen = Expectation of (D(G(A)) - 1)^2
        Otherwise: (NLL Loss)
            L_gen = Expectation of -log(D(G(A)))
        """
        if self.opt.use_ragan and use_ragan:
            loss = 0.5 * (tf.reduce_mean(tf.square(D(normal) - tf.reduce_mean(D(enhanced)))) + \
                          tf.reduce_mean(tf.squared_difference(D(enhanced) - tf.reduce_mean(D(normal)), 1.0)))
        elif self.opt.gan_mode == 'lsgan':
            loss = tf.reduce_mean(tf.squared_difference(D(enhanced), 1.0))
        elif self.opt.gan_mode == 'vanilla':
            loss = -1 * tf.reduce_mean(tf.log(D(enhanced) + eps))

        return loss

    def __perceptual_loss(low, enhanced, low_patch=None, enhanced_patch=None, low_patches=None, enhanced_patches=None):
        """
        Compute the self feature preserving loss on the low-light and enhanced image.
        """
        features_low = self.__vgg16_features(low)
        features_normal = self._vgg16_features(enhanced)

        if self.opt.patch_vgg:
            features_low_patch = self.__vgg16_features(low_patch)
            features_normal_patch = self.__vgg16_features(enhanced_patch)

        if self.opt.patchD_3 > 0:
            features_low_patches = self.__vgg16_features(low_patches)
            features_normal_patches = self.__vgg16_features(enhanced_patches)

        if self.opt.no_vgg_instance:
            loss = tf.reduce_mean(tf.squared_difference(features_low, features_normal))

            if self.opt.patch_vgg:
                loss += tf.reduce_mean(tf.squared_difference(features_low_patch,
                                                             features_normal_patch))

            if self.opt.patchD_3 > 0:
                loss += tf.reduce_mean(tf.squared_difference(features_low_patches,
                                                             features_normal_patches))
        else:
            loss = tf.reduce_mean(tf.squared_difference(instance_norm(features_low),
                                                        instance_norm(features_normal)))

            if self.opt.patch_vgg:
                loss += tf.reduce_mean(tf.squared_difference(instance_norm(features_low_patch),
                                                             instance_norm(features_normal_patch)))

            if self.opt.patchD_3 > 0:
                loss_ += tf.reduce_mean(tf.squared_difference(instance_norm(features_low_patches),
                                                              instance_norm(features_normal_patches)))

        return loss

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

    def __vgg16_features(image):
        """
        Extract features from image using VGG16 model.
        """
        vgg16_in = tf.keras.applications.vgg16.preprocess_input((image+1)*127.5)
        vgg16_features = self.vgg16_features(vgg16_in)

        return vgg16_features
