import math
import random

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras.losses import Loss
from tensorflow.python.ops import math_ops

from classifier_config import jpeg_paper, jpeg_paper_k12
from comp_classifier import ClassifierConfig

COLOR = 3

HDF5_NAME_X = "x_noise"
HDF5_NAME_Q = "y_quality"
LAST_MODEL_FILE_PATH = "save/last/"
FEATURE_MODEL_FILE_PATH = "save/feature_extractor/"

COMP_CLASSIFIER_CONFIGS = {
    "jpeg_paper":     jpeg_paper.JpegClassifier(),
    "jpeg_paper_k12": jpeg_paper_k12.JpegClassifier(),
}


class ConfidenceAwareMSE(Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        confidence_true = y_true[:, 0]
        confidence_pred = y_pred[:, 0]
        quality_true = y_true[:, 1]
        quality_pred = y_pred[:, 1]
        return math_ops.reduce_mean(math_ops.abs(confidence_true - confidence_pred) * 0.5
                                    + math_ops.abs(quality_true - quality_pred), axis=-1)


class DefaultClsSequence(Sequence):
    def __init__(self, data_path, batch_size, shuffle=False):
        hdf5 = h5py.File(data_path, 'r')
        self.X = hdf5[HDF5_NAME_X]
        self.Q = hdf5[HDF5_NAME_Q]
        self.batch_size = batch_size
        self.num_samples = len(self.X)
        self.shuffled_index_pool = list(range(self.num_samples))
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.shuffled_index_pool)

        print("processing", self.num_samples, "samples")

    # Number of batch in the Sequence
    def __len__(self):
        return math.ceil(self.num_samples / self.batch_size)

    # Gets batch at position `idx`.
    def __getitem__(self, idx):
        batch_x = []
        batch_y = []

        start = self.batch_size * idx
        end = min(start + self.batch_size, self.num_samples)
        idx_range = self.shuffled_index_pool[start:end]
        for i in idx_range:
            batch_x.append(self.X[i].astype('float32'))
            batch_y.append(self.Q[i].astype('uint8'))

        return np.array(batch_x), np.array(batch_y)

    # A callback called at the end of every epoch. However, TF2.1 doesn't call this due to a bug. TF2.2 is okay
    def on_epoch_end(self):
        # Shuffling data after each epoch ensures that we will not be "stuck" with too many bad batches
        if self.shuffle:
            random.shuffle(self.shuffled_index_pool)


class ConfClsSequence(Sequence):
    def __init__(self, config: ClassifierConfig, model, data_path, batch_size, shuffle=False):
        hdf5 = h5py.File(data_path, 'r')
        self.config = config
        self.model = model
        self.X = hdf5[HDF5_NAME_X]
        self.Q = hdf5[HDF5_NAME_Q]
        self.batch_size = batch_size
        self.num_samples = len(self.X)
        self.shuffled_index_pool = list(range(self.num_samples))
        self.shuffle = shuffle
        self.biggest_index = config.get_comp_qualities()[-1]
        random.shuffle(self.shuffled_index_pool)

        print("processing", self.num_samples, "samples")

    # Number of batch in the Sequence
    def __len__(self):
        return math.ceil(self.num_samples / self.batch_size)

    # Gets batch at position `idx`.
    def __getitem__(self, idx):
        batch_x = []
        batch_y = []

        start = self.batch_size * idx
        end = min(start + self.batch_size, self.num_samples)
        index_pool_subset = self.shuffled_index_pool[start:end]

        rand_batch_input = [self.X[rand_index].astype('float32') for rand_index in index_pool_subset]
        rand_batch_pred = self.model.predict(np.array(rand_batch_input))

        for sequential_idx, rand_index in enumerate(index_pool_subset):
            batch_x.append(self.X[rand_index].astype('float32'))

            quality_true = self.Q[rand_index].astype('float32')
            quality_pred = rand_batch_pred[sequential_idx][1]
            confidence_true = self.biggest_index - np.abs(quality_true - quality_pred)
            batch_y.append(np.array([confidence_true, quality_true]))  # change to 2nd dimension

        return np.array(batch_x), np.array(batch_y)

    # A callback called at the end of every epoch. However, TF2.1 doesn't call this due to a bug. TF2.2 is okay
    def on_epoch_end(self):
        # Shuffling data after each epoch ensures that we will not be "stuck" with too many bad batches
        if self.shuffle:
            random.shuffle(self.shuffled_index_pool)


def build_model(config: ClassifierConfig, trainable, name=None, batch_size=None, include_top=True):
    in_dim = config.get_input_dimension()
    input_shape = (in_dim, in_dim, COLOR)
    input_layer = Input(shape=input_shape, name=name, batch_size=batch_size)
    output_layer = config.stackup_layers(input_layer, include_top)
    return Model(input_layer, output_layer, trainable=trainable)


def get_feature_extractor(config: ClassifierConfig, entire_model, name=None, batch_size=None):
    feature_extractor = build_model(trainable=False, config=config, name=name, batch_size=batch_size, include_top=False)
    for idx, layer in enumerate(feature_extractor.layers):
        layer.set_weights(entire_model.layers[idx].get_weights())
    return feature_extractor


def get_classifier_config(comp_type) -> ClassifierConfig:
    config = COMP_CLASSIFIER_CONFIGS.get(comp_type)
    if not config:
        raise NotImplementedError("{} is not in {}".format(comp_type, COMP_CLASSIFIER_CONFIGS.keys()))
    return config
