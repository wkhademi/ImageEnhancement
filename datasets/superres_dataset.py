import tensorflow as tf
from utils import file_utils
from datasets.base_dataset import BaseDataset

AUTOTUNE = tf.data.experimental.AUTOTUNE

class SuperResDataset(BaseDataset):
    """
    Tensorflow dataset meant for loading in paired high-res/low-res images.

    Images are loaded from the path set by argument '--dir /path/to/data'
    """
    def __init__(self, opt, training):
        BaseDataset.__init__(self, opt, training)
        self.dir = opt.dir
        self.paths = file_utils.load_paths(self.dir)

    def generate(self, cache=True, shuffle_buffer_size=1000):
        dataset = tf.data.Dataset.from_tensor_slices(self.paths)

        if self.training:
            dataset = dataset.map(self._train_preprocess, num_parallel_calls=AUTOTUNE)
        else:
            dataset = dataset.map(self._test_preprocess, num_parallel_calls=AUTOTUNE)

        if cache:  # cache preprocessed images if set
            if isinstance(cache, str):
                dataset = dataset.cache(cache)
            else:
                dataset = dataset.cache()

        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.opt.batch_size)
        dataset = dataset.prefetch(AUTOTUNE)

        return dataset

    def _train_preprocess(self, path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=self.opt.channels)
        img = tf.image.random_crop(img, size=[self.opt.crop_size, self.opt.crop_size, self.opt.channels])
        img = tf.cast(img, dtype=tf.float32)
        highres = tf.image.random_flip_left_right(img)
        lowres = tf.image.resize(highres, [self.opt.scale_size, self.opt.scale_size])
        highres = (highres / 127.5) - 1.
        lowres = (lowres / 255.)

        return highres, lowres

    def _test_preprocess(self, path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=self.opt.channels)
        img = tf.cast(img, dtype=tf.float32)
        lowres = (img / 255.)

        return lowres
