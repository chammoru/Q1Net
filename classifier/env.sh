export PYTHONPATH="..:$PYTHONPATH"

# TensorFlow 2.16+ defaults to Keras 3; this project uses the Keras 2 API,
# so select the tf-keras (Keras 2) implementation.
export TF_USE_LEGACY_KERAS=1
