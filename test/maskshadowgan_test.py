import os
import sys
import tensorflow as tf
from test.base_test import BaseTest
from utils.file_utils import save_image
from models.maskshadowgan_model import MaskShadowGANModel
from options.maskshadowgan_options import MaskShadowGANOptions


class MaskShadowGANTest(BaseTest):
    """
    Tester for MaskShadowGAN model.
    """
    def __init__(self, opt):
        BaseTest.__init__(self, opt)

    def test(self):
        """
        Test the MaskShadowGAN model by loading in a saved model.
        """
        if self.opt.load_model is not None:
            checkpoint = 'checkpoints/' + self.opt.load_model
        else:
            print("Must load in a model to test on.")
            sys.exit(1)

        # build the CycleGAN graph
        graph = tf.Graph()
        with graph.as_default():
            maskshadowgan = MaskShadowGANModel(self.opt, training=False)
            dataA_iter, realA = maskshadowgan.generate_dataset()
            fakeImg = maskshadowgan.build()
            saver = tf.train.Saver()

        with tf.Session(graph=graph) as sess:
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint))  # restore graph and variables
            sess.run(dataA_iter.initializer)  # initialize dataset iterator
            samples_dir = os.path.expanduser(self.opt.sample_directory)

            for idx in range(self.opt.num_samples):
                real_image = sess.run(realA)  # fetch shadow image

                # generate shadow free image from shadow image
                generated_image = sess.run(fakeImg, feed_dict={maskshadowgan.realA: real_image})

                real_image_name = 'sampleA' + str(idx) + '.jpg'
                generated_image_name = 'generatedB' + str(idx) + '.jpg'

                # save real and generated image to samples directory
                save_image(real_image, os.path.join(samples_dir, real_image_name))
                save_image(generated_image, os.path.join(samples_dir, generated_image_name))


if __name__ == '__main__':
    parser = MaskShadowGANOptions(False)
    opt = parser.parse()
    tester = MaskShadowGANTest(opt)
    tester.test()
