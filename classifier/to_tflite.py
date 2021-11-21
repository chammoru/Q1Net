import argparse
from pathlib import Path

import class_core
from tf_util import to_tflite

TFLITE_MODEL_NAME = "comp_cls.tflite"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Compression Classifier model to tflite')
    parser.add_argument('--comp_type', required=True, type=str,
                        help='compression type such as jpeg or hevc')
    args = parser.parse_args()

    config = class_core.get_classifier_config(args.comp_type)

    out_path = Path(config.get_tflite_out_dir())
    out_path.mkdir(parents=True, exist_ok=True)

    # Convert to TensorFlow lite - force the input size
    batch_size = config.get_sample_per_1d() * config.get_sample_per_1d()
    cls_model = class_core.build_model(trainable=False, config=config, name="input_1", batch_size=batch_size)
    cls_model.load_weights(config.get_best_model_path()).expect_partial()

    to_tflite(cls_model, str(out_path / TFLITE_MODEL_NAME))
