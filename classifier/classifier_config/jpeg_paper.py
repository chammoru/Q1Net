from pathlib import PurePath

from tensorflow.keras.layers import Conv2D, Lambda, Dense, GlobalAveragePooling2D

import class_core
import tf_util
import util
from comp_classifier import ClassifierConfig

MODULE_NAME = PurePath(__file__).stem
BEST_MODEL_FILE_PATH = "./save/{}/best/".format(MODULE_NAME)
MAX_JPEG_QUALITY = 100
TARGET_JPEG_QUALITY = list(range(1, MAX_JPEG_QUALITY + 1))
K = 8
IN_DIM = 16
ALIGNMENT = 8


class JpegClassifier(ClassifierConfig):
    def get_best_model_path(self):
        return BEST_MODEL_FILE_PATH

    def get_comp_qualities(self):
        return TARGET_JPEG_QUALITY

    def get_best_quality(self):
        return MAX_JPEG_QUALITY

    def gen_comp(self, image, comp_quality):
        return util.generate_jpeg(image, comp_quality)

    def get_perturb_size(self):
        return self.get_block_size()

    def get_input_dimension(self):
        return IN_DIM

    def get_block_size(self):
        return ALIGNMENT

    def stackup_layers(self, x, include_top=True):
        x = tf_util.CBR(x, K, 3)
        x = tf_util.bottleneck(x, planes=K)
        x = tf_util.CBR(x, K * 2, 3)
        x = tf_util.bottleneck(x, planes=K * 2)
        x = tf_util.CBR(x, K * 4, 3)
        x = tf_util.bottleneck(x, planes=K * 4)
        x = tf_util.CBR(x, K * 8, 3)
        x = tf_util.bottleneck(x, planes=K * 8)
        x = tf_util.CBR(x, K * 4, 3)
        x = tf_util.bottleneck(x, planes=K * 4)
        x = tf_util.CBR(x, K * 2, 3)
        x = tf_util.bottleneck(x, planes=K * 2)
        x = Conv2D(K * 1, (3, 3), activation='relu')(x)
        x = GlobalAveragePooling2D()(x)
        if include_top:
            x = Dense(2, activation='sigmoid')(x)
            x = Lambda(lambda y: y * MAX_JPEG_QUALITY, name="results")(x)
        return x

    def get_tflite_out_dir(self):
        return "../android_demo/miraclefilterlib/src/main/assets/miraclefilter"

    def get_sample_per_1d(self):
        return 16

    def get_sequence(self, model, data_path, batch_size, shuffle=False):
        return class_core.ConfClsSequence(self, model, data_path, batch_size, shuffle)

    def use_confidence(self):
        return True

    def get_loss(self):
        return class_core.ConfidenceAwareMSE()

    def get_confidence_threshold(self):
        return 96  # TODO: need to grid-search
