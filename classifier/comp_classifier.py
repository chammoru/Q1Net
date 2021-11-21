from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence


class ClassifierConfig(metaclass=ABCMeta):
    @abstractmethod
    def get_best_model_path(self) -> str:
        """Return a path containing a model that performs best on
         validation dataset

        """
        raise NotImplementedError

    @abstractmethod
    def get_comp_qualities(self) -> list:
        """Return a list containing supported qualities by the target codec
        The list of quality should be in an ascending order so that get_comp_qualities()[-1] is the biggest number.

        """
        raise NotImplementedError

    @abstractmethod
    def get_best_quality(self):
        """Return a constant number representing the best quality for the target codec

        """
        raise NotImplementedError

    @abstractmethod
    def gen_comp(self, image, comp_quality) -> np.ndarray:
        """Generate and return a compressed version of a given image

        :param image: a numpy.array or path string which expresses a source image
        :param comp_quality: a int value specifies the compression quality
        """
        raise NotImplementedError

    @abstractmethod
    def get_perturb_size(self) -> int:
        """Return an int value that specifies how much perturbation should be
        applied when generating datasets

        """
        raise NotImplementedError

    @abstractmethod
    def get_input_dimension(self) -> int:
        """Return the input dimension of a network

        The input dimension should be larger than or equal to the alignment
        (block size). By using larger values, you may want to consider more
        pixels around the target region. This may allow the model to
        identify blocking artifacts more easily.
        """
        raise NotImplementedError

    @abstractmethod
    def get_block_size(self) -> int:
        """Return the block size for processing

        This block size refers to a basic processing unit. This also can be
        referred to alignment size.
        """
        raise NotImplementedError

    @abstractmethod
    def stackup_layers(self, input_layer: tf.Tensor, include_top=True) -> tf.Tensor:
        """The backbone network

        :param input_layer: The starting layer where a backbone network is
        built upon.
        :param include_top: whether to include the fully-connected layer at the top of the network.
        """
        raise NotImplementedError

    @abstractmethod
    def get_tflite_out_dir(self) -> str:
        """Return a path to store a generated tflite model

        """
        raise NotImplementedError

    @abstractmethod
    def get_sample_per_1d(self) -> int:
        """Return the number of sampling blocks in width or height

        """
        raise NotImplementedError

    @abstractmethod
    def get_sequence(self, model, data_path, batch_size, shuffle=False) -> Sequence:
        """Return a base object for fitting to a sequence of data, such as a dataset.

        """
        raise NotImplementedError

    @abstractmethod
    def use_confidence(self) -> bool:
        """Indicate whether the algorithm uses the confidence scheme

        """
        raise NotImplementedError

    # No @abstractmethod
    def get_confidence_threshold(self) -> float:
        """Get the confidence threshold acquired by Grid Search

        """
        raise NotImplementedError

    @abstractmethod
    def get_loss(self):
        """Return a loss (a.k.a. objective) function.

        """
        raise NotImplementedError
