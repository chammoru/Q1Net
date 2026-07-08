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
in_dim = config.get_input_dimension()
block = config.get_block_size()

print("Compression Qualities: {}".format(comp_qualities))

# Store patches one block larger than the network input, with the top-left
# corner aligned to the compression grid. The training sequence crops an
# (in_dim x in_dim) window at a random sub-block offset, so the model sees
# every possible drift between the patch and the codec's block grid. This
# replaces the old scheme that enumerated all block x block perturbations
# into the HDF5 file (a block^2 blow-up of near-duplicate samples).
store_dim = in_dim + block

num_patches = (args.num_samples + num_images - 1) // num_images
num_qualities = len(comp_qualities)
num_samples = num_images * num_patches * num_qualities

print("For each {} images, generate {} samples for each of {} qualities, yielding {} samples".
      format(num_images, num_patches, num_qualities, num_samples))

hdf5 = h5py.File(str(out_path / args.hdf5_name), 'w')
key_x = class_core.HDF5_NAME_X
key_q = class_core.HDF5_NAME_Q
hdf5.create_dataset(key_x, (num_samples, store_dim, store_dim, class_core.COLOR), dtype='uint8')
hdf5.create_dataset(key_q, (num_samples,), dtype='uint8')
X = hdf5[key_x]
Q = hdf5[key_q]

image_cache_dir = util.get_image_cache_dir(args.in_path, args.comp_type)

for image_file in image_files:
    # Compressed image path
    filename = PurePath(image_file).stem

    orig_image = cv2.imread(image_file)
    h1, w1 = orig_image.shape[0], orig_image.shape[1]

    comp_images = [util.get_cached_comp(config.gen_comp, filename, image_file, image_cache_dir, comp_quality)
                   for comp_quality in comp_qualities]

    # Create random grid-aligned patches (store_dim x store_dim) from each compressed image
    for i in range(num_patches):
        loop_limit = 3
        # Through the next loop, try to reduce the number of solid patch inputs
        while True:
            loop_limit -= 1
            h2 = block * random.randint(0, math.floor((h1 - store_dim) / block))
            w2 = block * random.randint(0, math.floor((w1 - store_dim) / block))

            orig_patch = orig_image[h2:h2 + store_dim, w2:w2 + store_dim].astype("uint8")
            variance = util.cal_variance(orig_patch)
            if variance > 0 or loop_limit <= 0:
                break

        if args.save_image:
            cv2.imwrite("{}_patch{:03d}_orig.png".format(filename, i), orig_patch)

        for quality_idx, comp_quality in enumerate(comp_qualities):  # Get compressed image with desired quality
            comp_image = comp_images[quality_idx]
            X[count_sample] = comp_image[h2:h2 + store_dim, w2:w2 + store_dim].astype("uint8")
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
