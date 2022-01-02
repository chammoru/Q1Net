# Q1Net: Quality Level Prediction of Image Compression using Block-wise Confidence-aware CNN - BMVC 2021
https://www.bmvc2021-virtualconference.com/conference/papers/paper_0813.html

## Authors
Kyuwon Kim (chammoru at gmail, q1.kim at samsung)  
Chulju Yang (ijn9429 at gmail, chulju at samsung)

## Citation
```
@InProceedings{kim2021q1net,
  title={Quality Level Prediction of Image Compression using Block-wise Confidence-aware CNN.},
  author={Kim, Kyuwon and Yang, Chulju},
  booktitle={Proceedings of the British Machine Vision Conference},
  month={Nov.},
  year={2021}
}
```

## Requirement
- TensorFlow >= 2.4

## Dataset
DIV2K dataset (https://data.vision.ee.ethz.ch/cvl/DIV2K/)

## Clone and setup
```bash
git clone https://github.com/chammoru/Q1Net.git

# Go to the source directory
cd Q1Net/classifier

# Setup environment
. ./env.sh
```

## Prediction Example
```bash
python3 ./predict_cls.py --in_path ../sample_image/monarch_jpeg_q20.png --comp_type jpeg_paper
```

## Evaluation
```bash
# Download dataset
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
unzip DIV2K_valid_HR.zip

python3 evaluate_cls.py --comp_type jpeg_paper --in_path DIV2K_valid_HR
```

## Training
```bash
# Download dataset
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
unzip DIV2K_train_HR.zip

sh batch_train_jpeg_paper.sh
```
In the `train.py`, `gen_data.py` creates a hdf5 file for training data:

## Convert model to tflite
```bash
python3 ./to_tflite.py --comp_type jpeg_paper
```

## Which apps can get benefits from Q1Net?
- Image/Photo Editor
- (Streaming) Video Player and Photo Viewer
- Web Browser
- Video Conferencing
- Instance Messaging App
- And many more
