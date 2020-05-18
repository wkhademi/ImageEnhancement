import os
import sys
import tensorflow as tf
from datetime import datetime
from utils.image_pool import ImagePool
from train.base_train import BaseTrain
from utils.mask_queue import MaskQueue, mask_generator
from models.maskshadowgan_model import MaskShadowGANModel
from options.maskshadowgan_options import MaskShadowGANOptions


class MaskShadowGANTrain(BaseTrain):
    """
    Trainer for MaskShadowGAN model.
    """
    def __init__(self, opt):
        BaseTrain.__init__(self, opt)

    def train(self):
        """
        Train the MaskShadowGAN model by starting from a saved checkpoint or from
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

        # build the Mask-ShadowGAN graph
        graph = tf.Graph()
        with graph.as_default():
            maskshadowgan = MaskShadowGANModel(self.opt, training=True)
            dataA_iter, dataB_iter, realA, realB = maskshadowgan.generate_dataset()
            fakeA, fakeB, optimizers, Gen_loss, D_A_loss, D_B_loss = maskshadowgan.build()
            saver = tf.train.Saver(max_to_keep=2)
            summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(checkpoint, graph)

        # create image pools for holding previously generated images
        fakeA_pool = ImagePool(self.opt.pool_size)
        fakeB_pool = ImagePool(self.opt.pool_size)

        # create queue to hold generated shadow masks
        mask_queue = MaskQueue(self.opt.queue_size)

        with tf.Session(graph=graph) as sess:
            if self.opt.load_model is not None:  # restore graph and variables
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint))
                ckpt = tf.train.get_checkpoint_state(checkpoint)
                step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
            else:
                sess.run(tf.global_variables_initializer())
                step = 0

            max_steps = self.opt.niter + self.opt.niter_decay

            # initialize data iterators
            sess.run([dataA_iter.initializer, dataB_iter.initializer])

            try:
                while step < max_steps:
                    try:
                        realA_img, realB_img = sess.run([realA, realB])  # fetch inputs

                        # generate shadow free image from shadow image
                        fakeB_img = sess.run(fakeB, feed_dict={maskshadowgan.realA: realA_img})

                        # generate shadow mask and add to mask queue
                        mask_queue.insert(mask_generator(realA_img, fakeB_img))
                        rand_mask = mask_queue.rand_item()

                        # generate shadow image from shadow free image and shadow mask
                        fakeA_img = sess.run(fakeA, feed_dict={maskshadowgan.realB: realB_img,
                                                               maskshadowgan.rand_mask: rand_mask})

                        # calculate losses for the generators and discriminators and minimize them
                        _, Gen_loss_val, D_B_loss_val, \
                        D_A_loss_val, sum = sess.run([optimizers, Gen_loss,
                                                      D_B_loss, D_A_loss, summary],
                                                      feed_dict={maskshadowgan.realA: realA_img,
                                                                 maskshadowgan.realB: realB_img,
                                                                 maskshadowgan.rand_mask: rand_mask,
                                                                 maskshadowgan.last_mask: mask_queue.last_item(),
                                                                 maskshadowgan.fakeA: fakeA_pool.query(fakeA_img),
                                                                 maskshadowgan.fakeB: fakeB_pool.query(fakeB_img)})

                        writer.add_summary(sum, step)
                        writer.flush()

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
                        sess.run([dataA_iter.initializer, dataB_iter.initializer])
            except KeyboardInterrupt:  # save training before exiting
                print("Saving models training progress to the `checkpoints` directory...")
                save_path = saver.save(sess, checkpoint + '/model.ckpt', global_step=step)
                print("Model saved as {}".format(save_path))
                sys.exit(0)


if __name__ == '__main__':
    parser = MaskShadowGANOptions(True)
    opt = parser.parse()
    trainer = MaskShadowGANTrain(opt)
    trainer.train()
