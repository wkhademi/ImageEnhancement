import os
import sys
import keras
import tensorflow as tf
from datetime import datetime
from keras import backend as K
from utils.image_pool import ImagePool
from train.base_train import BaseTrain
from models.enlightengan_model import EnlightenGANModel
from options.enlightengan_options import EnlightenGANOptions


class EnlightenGANTrain(BaseTrain):
    """
    Trainer for EnlightenGAN model.
    """
    def __init__(self, opt):
        BaseTrain.__init__(self, opt)

    def train(self):
        """
        Train the EnlightenGAN model by starting from a saved checkpoint or from
        the beginning.
        """
        if self.opt.load_model is not None:
            checkpoint = 'checkpoints/' + self.opt.load_model
        else:
            checkpoint_name = datetime.now().strftime("%d%m%Y-%H%M")
            checkpoint = 'checkpoints/{}'.format(checkpoint_name)

            try:
                os.makedirs(checkpoint)
            except os.error:
                print("Failed to make new checkpoint directory.")
                sys.exit(1)

        # build the EnlightenGAN graph
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                K.set_session(sess)

                enlightengan = EnlightenGANModel(self.opt, training=True)
                enlightengan.build()

                # feature extractor weights
                if self.opt.vgg:
                    vgg_weights = enlightengan.vgg16.get_weights()

                if self.opt.patch_vgg:
                    vgg_patch_weights = englightengan.vgg16_patch.get_weights()

                if self.opt.load_model is not None:  # restore graph and variables
                    saver.restore(sess, tf.train.latest_checkpoint(checkpoint))
                    ckpt = tf.train.get_checkpoint_state(checkpoint)
                    step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
                else:
                    step = 0
                    sess.run(tf.global_variables_initializer())

                    # hack for loading trained weights into TensorFlow graph
                    if self.opt.vgg:
                        enlightengan.vgg16.set_weights(vgg_weights)

                    if self.opt.patch_vgg:
                        enlightengan.vgg16_patch.set_weights(vgg_patch_weights)

if __name__ == '__main__':
    parser = EnlightenGANOptions(True)
    opt = parser.parse()
    trainer = EnlightenGANTrain(opt)
    trainer.train()
