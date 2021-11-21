import argparse
from pathlib import Path

import imageio
import rawpy

parser = argparse.ArgumentParser(description='DNG to PNG Convertor')
parser.add_argument('--in_path', required=True, type=str,
                    help='path to the source image')
args = parser.parse_args()

source = Path(args.in_path)
if not source.exists():
    print("Error: {} doest not exist".format(str(source)))
    exit(-1)

suffix = source.suffix
if suffix.lower() != ".dng":
    print("Error: {} is not the dng format".format(str(source)))
    exit(-2)

parent = source.parent
stem = source.stem

with rawpy.imread(str(source)) as raw:
    rgb = raw.postprocess()

imageio.imsave(str(parent / (stem + '.png')), rgb)

print("{} was converted".format(str(source)))
