import os
import sys
import tensorflow as tf
from datetime import datetime
from utils.image_pool import ImagePool
from train.base_train import BaseTrain
from models.CycleGAN_model import CycleGANModel
from options.cyclegan_options import CycleGANOptions


class CycleGANTrain(BaseTrain):
    """
    Trainer for CycleGAN model.
    """
    def __init__(self, opt):
        BaseTrain.__init__(self, opt)

    def train(self):
        """
        Train the CycleGAN model by starting from a saved checkpoint or from
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

        # create image pools for holding previously generated images
        fakeA_pool = ImagePool(self.opt.pool_size)
        fakeB_pool = ImagePool(self.opt.pool_size)

        # build the CycleGAN graph
        graph = tf.Graph()
        with graph.as_default():
            cyclegan = CycleGANModel(self.opt, training=True)
            fakeA, fakeB, optimizers, Gen_loss, D_A_loss, D_B_loss = cyclegan.build()
            saver = tf.train.Saver(max_to_keep=2)

        with tf.Session(graph=graph) as sess:
            if self.opt.load_model is not None: # restore graph and variables
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint))
                ckpt = tf.train.get_checkpoint_state(checkpoint)
                step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
            else:
                sess.run(tf.global_variables_initializer())
                step = 0

            max_steps = self.opt.niter + self.opt.niter_decay

            # initialize data iterators
            sess.run([cyclegan.dataA_iter.initializer, cyclegan.dataB_iter.initializer])

            try:
                while step < max_steps:
                    try:
                        fakeA_img, fakeB_img = sess.run([fakeA, fakeB])

                        # calculate losses for the generators and discriminators and minimize them
                        _, Gen_loss_val, D_B_loss_val, D_A_loss_val = sess.run([optimizers, Gen_loss,
                                                                               D_B_loss, D_A_loss],
                                                                               feed_dict={cyclegan.fakeA: fakeA_pool.query(fakeA_img),
                                                                                          cyclegan.fakeB: fakeB_pool.query(fakeB_img)})

                        # display the losses of the Generators and Discriminators
                        if step % self.opt.display_frequency == 0:
                            print('Step {}:'.format(step))
                            print('Gen_loss: {}'.format(Gen_loss_val))
                            print('D_B_loss: {}'.format(D_B_loss_val))
                            print('D_A_loss: {}'.format(D_A_loss_val))

                        # save a checkpoint of the model to the `checkpoints` directory
                        if step % self.opt.checkpoint_frequency == 0:
                            save_path = saver.save(sess, checkpoint + '/model.ckpt', global_step=step)
                            print("Model saved as {}".format(save_path))

                        step += 1
                    except tf.errors.OutOfRangeError:  # reinitializer iterators every full pass through dataset 
                        sess.run([cyclegan.dataA_iter.initializer, cyclegan.dataB_iter.initializer])
            except KeyboardInterrupt: # save training before exiting
                print("Saving models training progress to the `checkpoints` directory...")
                save_path = saver.save(sess, checkpoint + '/model.ckpt', global_step=step)
                print("Model saved as {}".format(save_path))
                sys.exit(0)


if __name__ == '__main__':
    parser = CycleGANOptions(True)
    opt = parser.parse()
    trainer = CycleGANTrain(opt)
    trainer.train()
