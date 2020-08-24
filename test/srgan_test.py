import os
import sys
import tensorflow as tf
from test.base_test import BaseTest
from utils.file_utils import save_image
from models.srgan_model import SRGANModel
from options.srgan_options import SRGANOptions


class SRGANTest(BaseTest):
    """
    Tester for SRGAN model.
    """
    def __init__(self, opt):
        BaseTest.__init__(self, opt)

    def test(self):
        """
        Test the SRGAN model by loading in a saved model.
        """
        if self.opt.load_model is not None:
            checkpoint = 'checkpoints/srgan/' + self.opt.load_model
        else:
            print("Must load in a model to test on.")
            sys.exit(1)

        # build the SRGAN graph
        graph = tf.Graph()
        with graph.as_default():
            srgan = SRGANModel(self.opt, training=False)
            superres = srgan.build()
            saver = tf.train.Saver()

        with tf.Session(graph=graph) as sess:
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint))  # restore graph and variables
            sess.run(srgan.data_iter.initializer)  # initialize dataset iterator
            samples_dir = os.path.expanduser(self.opt.sample_directory)

            for idx in range(self.opt.num_samples):
                lowres_image, superres_image = sess.run([srgan.lowres, superres])
                lowres_image_name = 'lowres' + str(idx) + '.jpg'
                superres_image_name = 'superres' + str(idx) + '.jpg'

                # save real and generated image to samples directory
                save_image(lowres_image, os.path.join(samples_dir, lowres_image_name))
                save_image(superres_image, os.path.join(samples_dir, superres_image_name))


if __name__ == '__main__':
    parser = SRGANOptions(False)
    opt = parser.parse()
    tester = SRGANTest(opt)
    tester.test()
