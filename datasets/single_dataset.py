import tensorflow as tf
from datasets.base_dataset import BaseDataset

AUTOTUNE = tf.data.experimental.AUTOTUNE

class SingleDataset(BaseDataset):
    """
    Tensorflow dataset meant for loading a single set of images.

    Images are loaded from the path set by argument '--dir /path/to/data'
    """
    def __init__(self, opt, training):
        BaseDataset.__init__(self, opt, training)
        self.dir = opt.dir
        self.paths = file_utils.load_paths(self.dir)

    def generate(self, cache=True, shuffle_buffer_size=1000):
        dataset = tf.data.Dataset.from_tensor_slices(self.paths)
        dataset = dataset.map(self._preprocess, num_parallel_calls=AUTOTUNE)

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

    def _preprocess(self, path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=self.opt.channels)
        img = tf.image.resize(img, [self.opt.scale_size, self.opt.scale_size],
                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        img = tf.image.random_crop(img, size=[self.opt.crop_size, self.opt.crop_size, self.opt.channels])
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = (img / 127.5) - 1.

        return img
