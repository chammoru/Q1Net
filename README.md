# Q1Net: Quality Level Prediction of Image Compression using Block-wise Confidence-aware CNN

Official implementation of the BMVC 2021 paper.
Paper: https://bmva-archive.org.uk/bmvc/2021/conference/papers/paper_0813.html

Q1Net predicts the quality level of a compressed image (e.g. the JPEG quality
factor) directly from the image, using a block-wise, confidence-aware CNN.

## Authors
- Kyuwon Kim (chammoru at gmail, q1.kim at samsung)
- Chulju Yang (ijn9429 at gmail, chulju at samsung)

## Citation
```bibtex
@InProceedings{kim2021q1net,
  title     = {Quality Level Prediction of Image Compression using Block-wise Confidence-aware CNN},
  author    = {Kim, Kyuwon and Yang, Chulju},
  booktitle = {Proceedings of the British Machine Vision Conference (BMVC)},
  month     = {November},
  year      = {2021}
}
```

## Requirements
- Python 3
- TensorFlow >= 2.4

## Dataset
This project uses the [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

## Clone and setup
```bash
git clone https://github.com/chammoru/Q1Net.git

# Go to the source directory
cd Q1Net/classifier

# Set up the environment (adds the repo root to PYTHONPATH)
. ./env.sh
```

The supported compression types (`--comp_type`) are `jpeg_paper` and `jpeg_paper_k12`.

## Prediction
Predict the quality level of a single image:
```bash
python3 ./predict_cls.py --in_path ../sample_image/monarch_jpeg_q20.png --comp_type jpeg_paper
```

## Evaluation
```bash
# Download the validation set
wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
unzip DIV2K_valid_HR.zip

python3 evaluate_cls.py --comp_type jpeg_paper --in_path DIV2K_valid_HR
```

## Training
```bash
# Download the training set
wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
unzip DIV2K_train_HR.zip

sh batch_train_jpeg_paper.sh
```
During training, `gen_data.py` generates an HDF5 file of training data that
`train.py` then consumes.

## Convert the model to TFLite
```bash
python3 ./to_tflite.py --comp_type jpeg_paper
```

## Applications
Q1Net can benefit a wide range of applications, including:
- Image/photo editors
- (Streaming) video players and photo viewers
- Web browsers
- Video conferencing
- Instant messaging apps
- And many more

## License
This code is released for **non-commercial research and evaluation purposes only**.
The methods implemented here are covered by U.S. Patent No. 12,462,356 B2, owned by
Samsung Electronics Co., Ltd.; **no patent license is granted**, and commercial use
requires a separate license. See [LICENSE](LICENSE) for the full terms.
