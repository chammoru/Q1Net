import argparse

import class_core

TFLITE_MODEL_NAME = "comp_cls.tflite"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Compression Classifier model to tflite')
    parser.add_argument('--comp_type', required=True, type=str,
                        help='compression type such as jpeg or hevc')
    args = parser.parse_args()

    config = class_core.get_classifier_config(args.comp_type)

    # Convert to TensorFlow lite - force the input size
    cls_model = class_core.build_model(trainable=False, config=config, name="input_1", batch_size=256)
    cls_model.load_weights(config.get_best_model_path())

    feature_extractor = class_core.get_feature_extractor(config, cls_model, name="input_1")
    feature_extractor.save_weights(class_core.FEATURE_MODEL_FILE_PATH)
