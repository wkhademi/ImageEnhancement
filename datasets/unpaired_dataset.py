import tensorflow as tf
from utils import file_utils
from datasets.base_dataset import BaseDataset

AUTOTUNE = tf.data.experimental.AUTOTUNE

class UnpairedDataset(BaseDataset):
    """
    Tensorflow dataset meant for loading in two unpaired sets of images.

    Image set A is loaded in from path set by argument '--dirA /path/to/dataA'
    Image set B is loaded in from path set by argument '--dirB /path/to/dataB'
    """
    def __init__(self, opt, training):
        BaseDataset.__init__(self, opt, training)
        self.dirA = opt.dirA
        self.dirB = opt.dirB
        self.pathsA = file_utils.load_paths(self.dirA)
        self.pathsB = file_utils.load_paths(self.dirB)

    def generate(self, cacheA=True, cacheB=True, shuffle_buffer_size=1000):
        datasetA = tf.data.Dataset.from_tensor_slices(self.pathsA)
        datasetA = datasetA.map(self._preprocess, num_parallel_calls=AUTOTUNE)

        if cacheA:  # cache preprocessed images if set
            if isinstance(cacheA, str):
                datasetA = datasetA.cache(cacheA)
            else:
                datasetA = datasetA.cache()

        datasetA = datasetA.shuffle(buffer_size=shuffle_buffer_size)
        datasetA = datasetA.repeat()
        datasetA = datasetA.batch(self.opt.batch_size)
        datasetA = datasetA.prefetch(AUTOTUNE)

        datasetB = tf.data.Dataset.from_tensor_slices(self.pathsB)
        datasetB = datasetB.map(self._preprocess, num_parallel_calls=AUTOTUNE)

        if cacheB:  # cache preprocessed images if set
            if isinstance(cacheB, str):
                datasetB = datasetB.cache(cacheB)
            else:
                datasetB = datasetB.cache()

        datasetB = datasetB.shuffle(buffer_size=shuffle_buffer_size)
        datasetB = datasetB.repeat()
        datasetB = datasetB.batch(self.opt.batch_size)
        datasetB = datasetB.prefetch(AUTOTUNE)

        return datasetA, datasetB

    def _preprocess(self, path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=self.opt.channels)
        img = tf.image.resize(img, [self.opt.scale_size, self.opt.scale_size],
                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        img = tf.image.random_crop(img, size=[self.opt.crop_size, self.opt.crop_size, self.opt.channels])
        img = tf.image.random_flip_left_right(img)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = (img / 127.5) - 1.

        return img
