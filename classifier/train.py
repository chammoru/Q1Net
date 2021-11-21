import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as keras_backend

import class_core
import evaluate_cls
import tf_util
import util

# Classifier model internal parameters
VAL_LOSS_FILE_NAME = "val_loss.txt"
STOP_SIGNAL_FILE_NAME = "./stop_training"

tf_util.allow_gpu_growth()


# noinspection PyBroadException
class Callbacks(tf.keras.callbacks.Callback):
    def __init__(self, model_path):
        super().__init__()
        self.best_val_loss = np.Inf
        self.model_path = model_path

        try:
            with open(str(Path(self.model_path) / VAL_LOSS_FILE_NAME), "r") as txt_file:
                best_val_loss = float(txt_file.readline())
                self.best_val_loss = best_val_loss
        except Exception:
            print("Error occurred when reading", VAL_LOSS_FILE_NAME)
            pass

        print("Initial best_val_loss is", self.best_val_loss)

    def on_epoch_end(self, epoch, logs=None):
        cur_val_loss = logs.get('val_loss')
        if np.less(cur_val_loss, self.best_val_loss):
            self.best_val_loss = cur_val_loss
            print("\n Save model for new best val_loss={}\n".format(cur_val_loss))
            # save the best model
            self.model.save_weights(self.model_path)
            try:
                with open(str(Path(self.model_path) / VAL_LOSS_FILE_NAME), "w") as val_loss_file:
                    val_loss_file.write(str(self.best_val_loss))
            except Exception:
                print("Error occurred when writing in", VAL_LOSS_FILE_NAME)
                pass

        if Path(STOP_SIGNAL_FILE_NAME).exists():
            print("\n Stop training")
            self.model.stop_training = True
            Path(STOP_SIGNAL_FILE_NAME).unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MiracleFilter model')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='the number of samples to be propagated through the network.')
    parser.add_argument('--out_path', type=str, default="out",
                        help='the path to save the results')
    parser.add_argument('--hdf5_train_path', required=True, type=str,
                        help='name of a hdf5 file for training')
    parser.add_argument('--base_weights', type=str, default="",
                        help='the base weights from which the training starts')
    parser.add_argument('--epochs', type=int, default=50,
                        help='the number of epochs to run')
    parser.add_argument('--hdf5_val_path', required=True, type=str,
                        help='name of a hdf5 file for validation')
    parser.add_argument('--steps_per_epoch', type=int,
                        help='(for debugging) the number of steps per epoch')
    parser.add_argument('--comp_type', required=True, type=str,
                        help='compression type such as jpeg or hevc')
    args = parser.parse_args()

    # Create output directory
    out_path = Path(args.out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    config = class_core.get_classifier_config(args.comp_type)
    comp_qualities = config.get_comp_qualities()
    best_model_path = config.get_best_model_path()

    # Create a model
    cls_model = class_core.build_model(config, trainable=True, name="input_1")
    if Path(args.base_weights).is_dir():
        print("Load weights from {}".format(args.base_weights))
        cls_model.load_weights(args.base_weights)
    else:
        print("Start training from scratch")
    cls_model.summary()
    cls_model.compile(optimizer='adam', loss=config.get_loss())

    # Prepare training dataset
    training_sequence = config.get_sequence(cls_model, args.hdf5_train_path, args.batch_size, True)

    # Prepare validation dataset
    validation_sequence = config.get_sequence(cls_model, args.hdf5_val_path, 32, False)

    # Train
    print("Start training for", args.epochs, "epoch")
    if args.steps_per_epoch is not None:
        steps_per_epoch = args.steps_per_epoch
    else:
        steps_per_epoch = len(training_sequence)

    try:
        history = cls_model.fit(training_sequence,
                                steps_per_epoch=steps_per_epoch,
                                validation_data=validation_sequence,
                                epochs=args.epochs,
                                callbacks=[Callbacks(best_model_path)])
    except KeyboardInterrupt:
        print("\n Stop training due to KeyboardInterrupt")
        history = cls_model.history

    cls_model.save_weights(class_core.LAST_MODEL_FILE_PATH)
    util.plot_graphs(history, str(out_path / "history_cls.png"))

    # Stop learning
    keras_backend.clear_session()

    # Do a test inference
    cls_model.load_weights(best_model_path)
    evaluate_cls.evaluate_patch(config, cls_model, validation_sequence, comp_qualities[-1])
