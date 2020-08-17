import os
import sys
import tensorflow as tf
from test.base_test import BaseTest
from utils.file_utils import save_image
from models.enlightengan_model import EnlightenGANModel
from options.enlightengan_options import EnlightenGANOptions


class EnlightenGANTest(BaseTest):
    """
    Tester for EnlightenGAN model.
    """
    def __init__(self, opt):
        BaseTest.__init__(self, opt)

    def test(self):
        """
        Test the EnlightenGAN model by loading in a saved model.
        """
        if self.opt.load_model is not None:
            checkpoint = 'checkpoints/enlightengan/' + self.opt.load_model
        else:
            print("Must load in a model to test on.")
            sys.exit(1)

        # build the CycleGAN graph
        graph = tf.Graph()
        with graph.as_default():
            enlightengan = EnlightenGANModel(self.opt, training=False)
            enhanced = enlightengan.build()
            saver = tf.train.Saver()

        with tf.Session(graph=graph) as sess:
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint))  # restore graph and variables
            sess.run(enlightengan.dataA_iter.initializer)  # initialize dataset iterator
            samples_dir = os.path.expanduser(self.opt.sample_directory)

            for idx in range(self.opt.num_samples):
                real_image, generated_image = sess.run([enlightengan.realA, enhanced])
                real_image_name = 'sampleA' + str(idx) + '.jpg'
                generated_image_name = 'generatedB' + str(idx) + '.jpg'

                # save real and generated image to samples directory
                save_image(real_image, os.path.join(samples_dir, real_image_name))
                save_image(generated_image, os.path.join(samples_dir, generated_image_name))


if __name__ == '__main__':
    parser = EnlightenGANOptions(False)
    opt = parser.parse()
    tester = EnlightenGANTest(opt)
    tester.test()
