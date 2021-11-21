import argparse
import sys
import time
from pathlib import PurePath, Path

import cv2
import numpy as np

import class_core
import util
from comp_classifier import ClassifierConfig

MINIMUM_VARIANCE = 1
COUNT_FOR_SAVING_PATCH = 0  # set a non-zero value (i.e. 10) to save input samples
PATH_FOR_SAVING_PATCH = Path("patches")
if COUNT_FOR_SAVING_PATCH > 0:
    PATH_FOR_SAVING_PATCH.mkdir(parents=True, exist_ok=True)


def predict_quality(model, image, config: ClassifierConfig):
    in_dim = config.get_input_dimension()
    block = config.get_block_size()
    h, w = image.shape[0], image.shape[1]
    in_stride = 3 * block

    sample_per_1d = config.get_sample_per_1d()
    aligned_h = (h // block) * block
    aligned_w = (w // block) * block

    batch_h = (aligned_h - in_stride) // block + 1
    batch_w = (aligned_w - in_stride) // block + 1

    interval = batch_h / (sample_per_1d + 1)
    indices_h = [int((i + 1) * interval) * block for i in range(sample_per_1d)]

    interval = batch_w / (sample_per_1d + 1)
    indices_w = [int((i + 1) * interval) * block for i in range(sample_per_1d)]

    in_rem = (block * 3 - in_dim) // 2
    patch_in_batch = [None] * (sample_per_1d * sample_per_1d)

    index = 0
    for i in indices_h:
        y = i + in_rem
        for j in indices_w:
            x = j + in_rem
            patch_in = image[y:y + in_dim, x:x + in_dim]
            patch_in = patch_in.astype('float32')
            patch_in_batch[index] = patch_in
            index += 1

    predicted = model.predict(np.array(patch_in_batch))
    extracted = []
    count = COUNT_FOR_SAVING_PATCH
    for i, patch_in in enumerate(patch_in_batch):
        if config.use_confidence():
            if count > 0:
                filename = "predC{:04d}_predQ{:04d}.png".format(int(predicted[i][0] * 10), int(predicted[i][1] * 10))
                filepath = str(PATH_FOR_SAVING_PATCH / filename)
                cv2.imwrite(filepath, patch_in)
                count -= 1

            if predicted[i][0] <= config.get_confidence_threshold():
                continue
            extracted.append(predicted[i][1])
        else:
            if config.get_sample_per_1d() > 1:  # Allow screen only when there are more than one sample.
                variance = util.cal_variance(patch_in)
                if variance <= MINIMUM_VARIANCE:
                    continue
            extracted.append(predicted[i])

    if not extracted:
        return config.get_best_quality()

    return np.median(extracted)


def predict_qualities_in_dir(model, in_path, comp_type):
    config = class_core.get_classifier_config(comp_type)

    image_files = util.get_file_list(in_path)
    image_files = sorted(image_files)
    pred_qualities = []

    file_count = 0

    for image_file in image_files:
        filename = PurePath(image_file).stem
        comp_image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        pred_quality = predict_quality(model, comp_image, config)
        pred_qualities.append(pred_quality)

        file_count += 1
        print("{:3d}/{}: {}, {:.1f}".format(file_count, len(image_files), filename, pred_quality))

    if file_count == 0:
        print("Warning: no file was predicted")
    else:
        print("Mean: {:.2f}".format(np.mean(pred_qualities)))


def main():
    parser = argparse.ArgumentParser(description='predict image compression quality'
                                     .format(PurePath(sys.argv[0]).name))
    parser.add_argument('--in_path', required=True, type=str,
                        help='path to read a source image or directory for images')
    parser.add_argument('--comp_type', required=True, type=str,
                        help='compression type such as jpeg or hevc')
    args = parser.parse_args()

    config = class_core.get_classifier_config(args.comp_type)
    best_model_path = config.get_best_model_path()
    cls_model = class_core.build_model(config, trainable=False)
    cls_model.load_weights(best_model_path)
    if Path(args.in_path).is_dir():
        predict_qualities_in_dir(cls_model, args.in_path, args.comp_type)
    else:
        compressed = cv2.imread(args.in_path, cv2.IMREAD_COLOR)
        # Preload model to measure only the processing time.
        predict_quality(cls_model, compressed, config)
        start = time.time()
        quality = predict_quality(cls_model, compressed, config)
        elapsed = time.time() - start
        print("predicted quality {:.2f}, estimated in {:.3f} seconds".format(quality, elapsed))

    return cls_model  # workaround the 'Unresolved object in checkpoint' warnings from TensorFlow


if __name__ == '__main__':
    _ = main()
