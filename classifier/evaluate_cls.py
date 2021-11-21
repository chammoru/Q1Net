import argparse
import itertools
import time
from pathlib import PurePath, Path

import numpy as np
import sklearn.metrics
from matplotlib import pyplot as plt

import class_core
import util
from comp_classifier import ClassifierConfig
from predict_cls import predict_quality


def evaluate_patch(config: ClassifierConfig, model, sequence, max_quality_idx, print_each=False):
    error_bins = [[] for _ in range(max_quality_idx + 1)]

    for idx, batch in enumerate(sequence):
        batch_x, batch_y = batch[0], batch[1]
        batch_y_pred = model.predict(batch_x)

        if config.use_confidence():
            true_qualities = batch_y[:, 1].astype('uint8')
            pred_qualities = batch_y_pred[:, 1]
        else:
            true_qualities = batch_y
            pred_qualities = np.squeeze(batch_y_pred)

        for true_y, pred_y in zip(true_qualities, pred_qualities):
            error_bins[true_y].append(true_y - pred_y)

            if print_each:
                print("true_y: {:3d}, pred_y: {:3.0f}, diff: {:2.0f}".
                      format(true_y, pred_y[0], true_y - pred_y))

    total_absolute_error = np.array([], dtype='float32')
    for q in range(max_quality_idx + 1):
        if len(error_bins[q]) <= 0:
            continue

        error = np.array(error_bins[q])
        absolute_error = np.abs(error)
        total_absolute_error = np.concatenate((total_absolute_error, absolute_error))
        print("quality {:3d}: mae {:2.0f}, bias {:2.0f}, std {:2.0f}".
              format(q, np.mean(absolute_error), -np.mean(error), np.std(error)))

    print("MAE: ", np.mean(total_absolute_error))


def save_confusion_matrix(cm, class_names, result_path, draw_text: bool = False):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
      :param cm: confusion matrix
      :param class_names: class names
      :param result_path:
      :param draw_text: show the exact value in the center of cells
    """
    plt.figure(figsize=(12, 9.5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    font_size = 6
    plt.xticks(tick_marks, class_names, fontsize=font_size, rotation=90)
    plt.yticks(tick_marks, class_names, fontsize=font_size)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    if draw_text:
        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(result_path)


def evaluate_images(model, in_path, out_path, comp_type):
    config = class_core.get_classifier_config(comp_type)
    comp_qualities = config.get_comp_qualities()

    image_files = util.get_file_list(in_path)
    true_qualities = []
    pred_qualities = []

    file_count = 0
    absolute_errors = []

    image_cache_dir = util.get_image_cache_dir(in_path, comp_type)
    for image_file in sorted(image_files):
        filename = PurePath(image_file).stem
        absolute_errors_in_image = []

        for true_quality in comp_qualities:
            comp_image = util.get_cached_comp(config.gen_comp,
                                              filename, image_file, image_cache_dir, true_quality)

            pred_quality = predict_quality(model, comp_image, config)
            ae = abs(true_quality - pred_quality)
            absolute_errors.append(ae)
            absolute_errors_in_image.append(ae)
            true_qualities.append(true_quality)
            pred_qualities.append(pred_quality)

        file_count += 1
        print("Images processed: {:3d}/{}: {:.2f}".
              format(file_count, len(image_files), np.mean(absolute_errors_in_image)))

    if file_count == 0:
        print("Warning: no file was predicted")
    else:
        print("mae {:.2f}, std of error {:.2f}".format(np.mean(absolute_errors), np.std(absolute_errors)))

    pred_qualities = np.clip(pred_qualities, comp_qualities[0], comp_qualities[-1])
    confusion_matrix = sklearn.metrics.confusion_matrix(true_qualities, np.around(pred_qualities))
    class_names = [str(q) for q in comp_qualities]
    save_confusion_matrix(confusion_matrix, class_names, str(Path(out_path) / "confusion_matrix.png"))


def main():
    parser = argparse.ArgumentParser(description='Evaluate Classifier model')
    parser.add_argument('--hdf5_val_path', type=str, default=None,
                        help='name of a hdf5 file for validation')
    parser.add_argument('--comp_type', required=True, type=str,
                        help='compression type such as jpeg or hevc')
    parser.add_argument('--in_path', type=str, default="",
                        help='directory path storing images')
    parser.add_argument('--out_path', type=str, default="out",
                        help='path to save the results')
    args = parser.parse_args()

    Path(args.out_path).mkdir(parents=True, exist_ok=True)

    config = class_core.get_classifier_config(args.comp_type)
    comp_qualities = config.get_comp_qualities()
    best_model_path = config.get_best_model_path()
    cls_model = class_core.build_model(config, trainable=False)
    cls_model.load_weights(best_model_path).expect_partial()

    if args.hdf5_val_path is not None:
        print("Start evaluating patches")
        # Prepare validation set
        validation_sequence = config.get_sequence(cls_model, args.hdf5_val_path, 64, False)

        start = time.time()
        evaluate_patch(config, cls_model, validation_sequence, comp_qualities[-1])
        elapsed = time.time() - start
        print("It took {:.3f} seconds".format(elapsed))
    elif args.in_path and Path(args.in_path).is_dir():
        print("Start evaluating images")
        evaluate_images(cls_model, args.in_path, args.out_path, args.comp_type)
    else:
        raise RuntimeError("You should enter either in_path or hdf5_val_path")

    return cls_model


if __name__ == '__main__':
    _ = main()
