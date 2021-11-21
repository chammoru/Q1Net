python gen_data.py --in_path ./DIV2K_train_HR --num_samples 800 --hdf5_name cls_train_jpeg_paper.h5 --comp_type jpeg_paper || exit 1
python gen_data.py --in_path ./DIV2K_valid_HR --num_samples  100 --hdf5_name cls_val_jpeg.h5 --comp_type jpeg_paper || exit 1
python train.py --hdf5_train_path cls_train_jpeg_paper.h5 --hdf5_val_path cls_val_jpeg.h5 --base_weights ./save/jpeg_paper/best/ --comp_type jpeg_paper || exit 1
python gen_data.py --in_path ./DIV2K_train_HR --num_samples 800 --hdf5_name cls_train_jpeg_paper.h5 --comp_type jpeg_paper || exit 1
python train.py --hdf5_train_path cls_train_jpeg_paper.h5 --hdf5_val_path cls_val_jpeg.h5 --base_weights ./save/jpeg_paper/best/ --comp_type jpeg_paper --batch_size 4096 || exit 1
python gen_data.py --in_path ./DIV2K_train_HR --num_samples 800 --hdf5_name cls_train_jpeg_paper.h5 --comp_type jpeg_paper || exit 1
python train.py --hdf5_train_path cls_train_jpeg_paper.h5 --hdf5_val_path cls_val_jpeg.h5 --base_weights ./save/jpeg_paper/best/ --comp_type jpeg_paper --batch_size 4096 || exit 1
