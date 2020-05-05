import os
import sys
import tensorflow as tf
from test.base_test import BaseTest
from utils.file_utils import save_image
from models.CycleGAN_model import CycleGANModel
from options.cyclegan_options import CycleGANOptions


class CycleGANTest(BaseTest):
    """
    Tester for CycleGAN model.
    """
    def __init__(self, opt):
        BaseTest.__init__(self, opt)

    def test(self):
        """
        Test the CycleGAN model by loading in a saved model.
        """
        if self.opt.load_model is not None:
            checkpoint = 'checkpoints/' + self.opt.load_model
        else:
            print("Must load in a model to test on.")
            sys.exit(1)

        # build the CycleGAN graph
        graph = tf.Graph()
        with graph.as_default():
            cyclegan = CycleGANModel(self.opt, training=False)
            fakeImg = cyclegan.build()
            saver = tf.train.Saver()

        with tf.Session(graph=graph) as sess:
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint))  # restore graph and variables
            sess.run(cyclegan.dataA_iter.initializer)  # initialize dataset iterator
            samples_dir = os.path.expanduser(self.opt.sample_directory)

            for idx in range(self.opt.num_samples):
                generated_image = sess.run(fakeImg)
                image_name = 'sample' + str(idx) + '.jpg'
                save_image(generated_image, os.path.join(samples_dir, image_name))


if __name__ == '__main__':
    parser = CycleGANOptions(False)
    opt = parser.parse()
    tester = CycleGANTest(opt)
    tester.test()
