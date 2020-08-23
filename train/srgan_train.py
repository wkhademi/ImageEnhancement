import os
import sys
import keras
import tensorflow as tf
from datetime import datetime
from keras import backend as K
from train.base_train import BaseTrain
from models.srgan_model import SRGANModel
from options.srgan_options import SRGANOptions


class SRGANTrain(BaseTrain):
    """
    Trainer for SRGAN model.
    """
    def __init__(self, opt):
        BaseTrain.__init__(self, opt)

    def train(self):
        """
        Train the SRGAN model by starting from a saved checkpoint or from
        the beginning.
        """
        if self.opt.load_model is not None:
            checkpoint = 'checkpoints/srgan/' + self.opt.load_model
        else:
            checkpoint_name = datetime.now().strftime("%d%m%Y-%H%M")
            checkpoint = 'checkpoints/srgan/{}'.format(checkpoint_name)

            try:
                os.makedirs(checkpoint)
            except os.error:
                print("Failed to make new checkpoint directory.")
                sys.exit(1)

        # build the SRGAN graph
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                K.set_session(sess)

                srgan = SRGANModel(self.opt, training=True)
                gen_optimizer, optimizers, Gen_init_loss, Gen_loss, D_loss = srgan.build()
                saver = tf.train.Saver(max_to_keep=2)
                summary = tf.summary.merge_all()
                writer = tf.summary.FileWriter(checkpoint, graph)

                # feature extractor weights
                vgg_weights = srgan.vgg19.get_weights()

                if self.opt.load_model is not None:  # restore graph and variables
                    saver.restore(sess, tf.train.latest_checkpoint(checkpoint))
                    ckpt = tf.train.get_checkpoint_state(checkpoint)
                    step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
                else:
                    step = 0
                    sess.run(tf.global_variables_initializer())

                # hack for loading trained weights into TensorFlow graph
                srgan.vgg19.set_weights(vgg_weights)

                # initialize data iterator
                sess.run([srgan.data_iter.initializer])

                start_epoch = (step*self.opt.batch_size) / len(srgan.dataset.paths)

                try:
                    if start_epoch < self.opt.num_epochs_init:
                        # initialize generator
                        for epoch in range(start_epoch, self.opt.num_epochs_init):
                            while True:
                                try:
                                    _, Gen_loss_val, sum = sess.run([gen_optimizer, Gen_init_loss, summary])

                                    writer.add_summary(sum, step)
                                    writer.flush()

                                    # display the losses of the Generators
                                    if step % self.opt.display_frequency == 0:
                                        print('Epoch {} - Step {}:'.format(epoch, step))
                                        print('Gen_init_loss: {}'.format(Gen_loss_val))

                                    # save a checkpoint of the model to the `checkpoints` directory
                                    if step % self.opt.checkpoint_frequency == 0:
                                        save_path = saver.save(sess, checkpoint + '/model.ckpt', global_step=step)
                                        print("Model saved as {}".format(save_path))

                                    step += 1
                                except tf.errors.OutOfRangeError:  # reinitialize iterator every epoch
                                    sess.run([srgan.data_iter.initializer])
                                    break
                    else:
                        start_epoch = start_epoch - self.opt.num_epochs_init

                    # train generator and discriminator
                    for epoch in range(start_epoch, self.opt.num_epochs):
                        while True:
                            try:
                                _, Gen_loss_val, D_loss_val, sum = sess.run([optimizers, Gen_loss, D_loss, summary])

                                writer.add_summary(sum, step)
                                writer.flush()

                                # display the losses of the Generators
                                if step % self.opt.display_frequency == 0:
                                    print('Epoch {} - Step {}:'.format(epoch, step))
                                    print('Gen_loss: {}'.format(Gen_loss_val))
                                    print('D_loss: {}'.format(D_loss_val))

                                # save a checkpoint of the model to the `checkpoints` directory
                                if step % self.opt.checkpoint_frequency == 0:
                                    save_path = saver.save(sess, checkpoint + '/model.ckpt', global_step=step)
                                    print("Model saved as {}".format(save_path))

                                step += 1
                            except tf.errors.OutOfRangeError:  # reinitialize iterator every epoch
                                sess.run([srgan.data_iter.initializer])
                                break
                except KeyboardInterrupt:  # save training before exiting
                    print("Saving models training progress to the `checkpoints` directory...")
                    save_path = saver.save(sess, checkpoint + '/model.ckpt', global_step=step)
                    print("Model saved as {}".format(save_path))
                    sys.exit(0)


if __name__ == '__main__':
    parser = SRGANOptions(True)
    opt = parser.parse()
    trainer = SRGANTrain(opt)
    trainer.train()
