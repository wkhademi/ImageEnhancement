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
            checkpoint = 'checkpoints/enlightengan/' + self.opt.load_model
        else:
            checkpoint_name = datetime.now().strftime("%d%m%Y-%H%M")
            checkpoint = 'checkpoints/enlightengan/{}'.format(checkpoint_name)

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
                enhanced, optimizers, Gen_loss, D_loss, D_P_loss = enlightengan.build()
                saver = tf.train.Saver(max_to_keep=2)
                summary = tf.summary.merge_all()
                writer = tf.summary.FileWriter(checkpoint, graph)

                # feature extractor weights
                if self.opt.vgg:
                    vgg_weights = enlightengan.vgg16.get_weights()

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

                max_steps = self.opt.niter + self.opt.niter_decay

                # initialize data iterators
                sess.run([enlightengan.dataA_iter.initializer, enlightengan.dataB_iter.initializer])

                try:
                    while step < max_steps:
                        try:
                            # calculate losses for the generators and discriminators and minimize them
                            _, Gen_loss_val, D_loss_val, \
                            D_P_loss_val, sum = sess.run([optimizers, Gen_loss,
                                                          D_loss, D_P_loss, summary])

                            writer.add_summary(sum, step)
                            writer.flush()

                            # display the losses of the Generators and Discriminators
                            if step % self.opt.display_frequency == 0:
                                print('Step {}:'.format(step))
                                print('Gen_loss: {}'.format(Gen_loss_val))
                                print('D_loss: {}'.format(D_loss_val))
                                print('D_P_loss: {}'.format(D_P_loss_val))

                            # save a checkpoint of the model to the `checkpoints` directory
                            if step % self.opt.checkpoint_frequency == 0:
                                save_path = saver.save(sess, checkpoint + '/model.ckpt', global_step=step)
                                print("Model saved as {}".format(save_path))

                            step += 1
                        except tf.errors.OutOfRangeError:  # reinitializer iterators every full pass through dataset
                            sess.run([enlightengan.dataA_iter.initializer, enlightengan.dataB_iter.initializer])
                except KeyboardInterrupt: # save training before exiting
                    print("Saving models training progress to the `checkpoints` directory...")
                    save_path = saver.save(sess, checkpoint + '/model.ckpt', global_step=step)
                    print("Model saved as {}".format(save_path))
                    sys.exit(0)


if __name__ == '__main__':
    parser = EnlightenGANOptions(True)
    opt = parser.parse()
    trainer = EnlightenGANTrain(opt)
    trainer.train()
