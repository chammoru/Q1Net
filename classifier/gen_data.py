# Script to generate training image set:
#
# Input (in_path) is the folder with the arbitrary image (*.jpg|*.png) set.

import argparse
import math
import random
from pathlib import PurePath, Path

import cv2
import h5py

import class_core
import util

parser = argparse.ArgumentParser(description='Low Quality Compression Image Generator')
parser.add_argument('--in_path', default='.', type=str,
                    help='path to the source image dataset (default: current dir)')
parser.add_argument('--out_path', default='.', type=str,
                    help='path to save generated images (default: current dir)')
parser.add_argument('--num_samples', default=1000000, type=int,
                    help='number of approximately desired samples for each compression quality (default: 1000000)')
parser.add_argument('--hdf5_name', required=True, type=str,
                    help='name of a hdf5 file that will be generated')
parser.add_argument('--comp_type', required=True, type=str,
                    help='compression type such as jpeg or hevc')
parser.add_argument('--save_image', default=False, action='store_true',
                    help='save png images for debugging')
parser.add_argument('--perturb', default=None, type=int,
                    help='int value that specifies how much perturbation should be applied when generating datasets')
args = parser.parse_args()

out_path = Path(args.out_path)
out_path.mkdir(parents=True, exist_ok=True)

print("Process files in:", args.in_path)
image_files = util.iglob(args.in_path, ('.jpg', '.jpeg', '.png'))

count_sample = 0
count_image = 0
num_images = len(image_files)
if num_images <= 0:
    print("There is no images in the directory[{}]".format(args.in_path))
    exit(1)

config = class_core.get_classifier_config(args.comp_type)
comp_qualities = config.get_comp_qualities()
if args.perturb is None:
    perturb = config.get_perturb_size()
else:
    perturb = args.perturb
in_dim = config.get_input_dimension()
block = config.get_block_size()

print("Compression Qualities: {}".format(comp_qualities))

in_side = (in_dim - block) // 2
in_stride = max(((in_dim + block - 1) // block) * block, block * 3)
in_pad = (in_stride - in_dim) // 2

num_patches = (args.num_samples + num_images - 1) // num_images
num_qualities = len(comp_qualities)
num_samples = num_images * num_patches * num_qualities * perturb * perturb

print("For each {} images, generate {} samples for each of {} qualities, yielding {} samples".
      format(num_images, num_patches, num_qualities, num_samples))

hdf5 = h5py.File(str(out_path / args.hdf5_name), 'w')
key_x = class_core.HDF5_NAME_X
key_q = class_core.HDF5_NAME_Q
hdf5.create_dataset(key_x, (num_samples, in_dim, in_dim, class_core.COLOR), dtype='uint8')
hdf5.create_dataset(key_q, (num_samples,), dtype='uint8')
X = hdf5[key_x]
Q = hdf5[key_q]

image_cache_dir = util.get_image_cache_dir(args.in_path, args.comp_type)
perturb_list = []
for h_perturb in range(-(perturb // 2), math.ceil(perturb / 2)):
    for w_perturb in range(-(perturb // 2), math.ceil(perturb / 2)):
        perturb_list.append((h_perturb, w_perturb))

for image_file in image_files:
    # Compressed image path
    filename = PurePath(image_file).stem

    orig_image = cv2.imread(image_file)
    h1, w1 = orig_image.shape[0], orig_image.shape[1]

    comp_images = [util.get_cached_comp(config.gen_comp, filename, image_file, image_cache_dir, comp_quality)
                   for comp_quality in comp_qualities]

    # Create random image patches (24x24) from compressed image, and matching 12x12 center from uncompressed one
    for i in range(num_patches):
        loop_limit = 3
        # Through the next loop, try to reduce the number of solid patch inputs
        while True:
            loop_limit -= 1
            h2 = block * random.randint(0, math.floor((h1 - in_stride) / block)) + in_pad
            w2 = block * random.randint(0, math.floor((w1 - in_stride) / block)) + in_pad

            orig_patch = orig_image[h2:h2 + in_dim, w2:w2 + in_dim].astype("uint8")
            variance = util.cal_variance(orig_patch)
            if variance > 0 or loop_limit <= 0:
                break

        if args.save_image:
            cv2.imwrite("{}_patch{:03d}_orig.png".format(filename, i), orig_patch)

        for (h_perturb, w_perturb) in perturb_list:
            h3 = h2 + h_perturb
            w3 = w2 + w_perturb

            for quality_idx, comp_quality in enumerate(comp_qualities):  # Get compressed image with desired quality
                comp_image = comp_images[quality_idx]
                X[count_sample] = comp_image[h3:h3 + in_dim, w3:w3 + in_dim].astype("uint8")
                Q[count_sample] = comp_quality

                if args.save_image:
                    cv2.imwrite("{}_patch{:03d}_quality_{}.png".format(filename, i, comp_quality), X[count_sample])

                count_sample += 1

    count_image += 1
    print("Processed {}/{} image".format(count_image, num_images), end='\r')

hdf5.close()

if count_sample != num_samples:
    print("Error: The HDFS file might be inappropriate!: expected({}) vs actual({})".format(num_samples, count_sample))
else:
    print("Finished generating data for classification")
