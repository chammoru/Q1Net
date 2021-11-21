import random
import string
import subprocess
from pathlib import Path, PurePath
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage

plt.switch_backend('agg')

VIDEO_CONFIGS = {
    "h264": {"encoders": "libx264", "params": "-x264-params"},
    "hevc": {"encoders": "libx265", "params": "-x265-params"},
    "hevc_confidence": {"encoders": "libx265", "params": "-x265-params"},
}


def get_file_list(dataset_path, pattern="*.*"):
    dataset_path = Path(dataset_path)
    paths = [str(path) for path in dataset_path.glob(pattern)]
    if len(paths) == 0:
        print("Error: couldn't find the directory", dataset_path)
        exit(-1)

    return paths


def plot_graphs(history, file_path):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    num_epochs = len(loss)
    epochs = range(num_epochs)  # Get number of epochs

    plt.figure(figsize=(18, 9))

    # ------------------------------------------------
    # Plot train loss and val loss
    # ------------------------------------------------
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # ------------------------------------------------
    # Plot zoomed train loss and val loss
    # ------------------------------------------------
    plt.subplot(1, 2, 2)
    cut = num_epochs * 2 // 3
    plt.plot(epochs[cut:], loss[cut:], label='Training Loss')
    plt.plot(epochs[cut:], val_loss[cut:], label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    if file_path:
        plt.savefig(file_path)
    else:
        plt.show()


def generate_jpeg(image, jpeg_quality):
    if isinstance(image, str):
        image = cv2.imread(image)
    _, enc_img = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    return cv2.imdecode(enc_img, 1).astype('float32')


def generate_webp(image, webp_quality):
    if isinstance(image, str):
        image = cv2.imread(image)
    _, enc_img = cv2.imencode('.webp', image, [int(cv2.IMWRITE_WEBP_QUALITY), webp_quality])
    return cv2.imdecode(enc_img, 1).astype('float32')


def nonce_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


# Recommended x265 version: https://packages.debian.org/sid/libx265-192
def generate_video(image, crf, codec):
    tmp_video_path = "./temp_video_{}.mp4".format(nonce_generator())
    tmp_png_path = "./temp_video_{}.png".format(nonce_generator())

    if isinstance(image, str):
        path = image
        image = cv2.imread(path)
    else:
        path = tmp_png_path
        cv2.imwrite(path, image)

    code = subprocess.call(['ffmpeg', '-y', '-hide_banner', '-nostdin',
                            '-loglevel', 'error',
                            '-i', path,
                            '-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2',
                            '-c:v', VIDEO_CONFIGS[codec].get("encoders"),
                            '-crf', str(crf),
                            '-pix_fmt', 'nv12',
                            VIDEO_CONFIGS[codec].get("params"), 'log-level=error',  # 'no-deblock=true' was used before
                            tmp_video_path])
    if code != 0:
        print("Failed to convert image to " + codec)
        exit(code)

    cap = cv2.VideoCapture(tmp_video_path)
    _, frame = cap.read()

    if Path(tmp_png_path).exists():
        Path(tmp_png_path).unlink()
    Path(tmp_video_path).unlink()

    h, w, _ = frame.shape
    video = image.astype('float32')
    video[:h, :w] = frame.astype('float32')  # frame may be cropped, so we deliver an image with the original size

    return video


# noinspection PyUnusedLocal
def generate_gif(image, quality):
    tmp_gif_path = "./temp_q1kim_{}.gif".format(nonce_generator())
    tmp_png_path = "./temp_q1kim_{}.png".format(nonce_generator())

    if isinstance(image, str):
        path = image
        image = cv2.imread(path)
    else:
        path = tmp_png_path
        cv2.imwrite(path, image)

    code = subprocess.call(['ffmpeg', '-y', '-hide_banner', '-nostdin',
                            '-loglevel', 'error',
                            '-r', '1',
                            '-i', path,
                            tmp_gif_path])
    if code != 0:
        print("Failed to convert image to gif")
        exit(code)

    cap = cv2.VideoCapture(tmp_gif_path)
    _, gif = cap.read()

    if Path(tmp_png_path).exists():
        Path(tmp_png_path).unlink()
    Path(tmp_gif_path).unlink()

    return gif.astype('float32')


def get_cached_comp(gen_comp, filename, path, image_cache_dir, comp_quality):
    image_cache_dir = Path(image_cache_dir)
    if image_cache_dir.exists():
        image_cache_file = "{}_quality_{:02d}.png".format(filename, comp_quality)
        image_cache_path = str(image_cache_dir / image_cache_file)
        img = cv2.imread(image_cache_path)
        if img is None:
            print("\nWarning: Cannot find {}".format(image_cache_file), end="")
            comp_image = gen_comp(path, comp_quality)
            cv2.imwrite(image_cache_path, comp_image)
            print(", regenerated at '{}'".format(image_cache_path))
        else:
            comp_image = img.astype('float32')
    else:
        comp_image = gen_comp(path, comp_quality)
    return comp_image


def is_grayscale(i):
    if (i[:, :, 0] - i[:, :, 1] == 0).all() and (i[:, :, 1] - i[:, :, 2] == 0).all():
        return True
    else:
        return False


def gaussian_noise(img, scale=10.0):
    gauss = np.random.normal(0, scale, img.size).astype('float32')
    gauss = gauss.reshape((img.shape[0], img.shape[1], img.shape[2]))
    # Add the Gaussian noise to the image
    img_gauss = img.astype("float32") + gauss
    img_gauss = np.clip(img_gauss, 0, 255)
    return img_gauss


def gaussian_noise_y(img, scale=10.0):
    img = img.astype('float32')
    y_cr_cb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = y_cr_cb[:, :, 0]
    noisy_y = gaussian_noise(y[..., np.newaxis], scale)
    y_cr_cb[:, :, 0] = noisy_y[:, :, 0]
    recon = cv2.cvtColor(y_cr_cb, cv2.COLOR_YCrCb2BGR)
    return np.clip(recon, 0, 255)


def salt_and_pepper_noise(img, amount=0.05):
    s_vs_p = 0.5
    out = np.copy(img)
    # Salt mode
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coordinates = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    out[tuple(coordinates)] = 255
    # Pepper mode
    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
    coordinates = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    out[tuple(coordinates)] = 0

    return out.astype("float32")


def poisson_noise(img):
    num_uniques = len(np.unique(img))
    val = 2 ** np.ceil(np.log2(num_uniques))
    noisy = np.random.poisson(img * val) / float(val)
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype("float32")


def speckle_noise(image):
    row, col, ch = image.shape
    gauss = np.random.randn(row, col, ch)
    gauss = gauss.reshape((row, col, ch))
    noisy = image + image * gauss
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype("float32")


def sharpen_image(image):
    amount = 0.2
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255)


def get_image_cache_dir(dir_in, comp_type):
    dir_in = dir_in.rstrip("/").rstrip("\\")
    image_cache_dir = dir_in + "_" + comp_type
    return image_cache_dir


def iglob(search_path, exts) -> List[str]:  # case insensitive
    file_list = []
    for file in Path(search_path).glob('*.*'):
        ext = PurePath(str(file)).suffix.lower()
        if ext in exts:
            file_list.append(str(file))
    return file_list


def irglob(search_path, exts) -> List[str]:  # case insensitive and recursive
    file_list = []
    for file in Path(search_path).rglob('*.*'):
        ext = PurePath(str(file)).suffix.lower()
        if ext in exts:
            file_list.append(str(file))
    return file_list


def cal_variance(image):
    max_val_by_sobel_kernel = 1020
    # https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x, alpha=(255 / max_val_by_sobel_kernel))
    abs_grad_y = cv2.convertScaleAbs(grad_y, alpha=(255 / max_val_by_sobel_kernel))
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return np.mean(grad)


def random_horizontal_flip(in_img, gt_img):
    rn = random.uniform(0, 1)
    if rn < 0.5:
        return in_img, gt_img
    else:
        return cv2.flip(in_img, 1), cv2.flip(gt_img, 1)


def random_rotate(in_img, gt_img):
    valid_rotations = [cv2.ROTATE_90_CLOCKWISE,
                       cv2.ROTATE_180,
                       cv2.ROTATE_90_COUNTERCLOCKWISE]
    rotate_0 = -1
    rn = random.choice(valid_rotations + [rotate_0])
    if rn in valid_rotations:
        return cv2.rotate(in_img, rn), cv2.rotate(gt_img, rn)
    else:
        return in_img, gt_img


def crop_for_pixel_alignment(image, multiple):
    if multiple == 1:  # In many cases 'alignment' is 1, so it's worth checking whether 'multiple' is 1
        return image

    h, w = image.shape[0], image.shape[1]
    h = (h // multiple) * multiple
    w = (w // multiple) * multiple
    return image[:h, :w]


def stack_quality(input_patch, row, column, quality):
    quality_patch = np.full((row, column, 1), quality, dtype=np.float32)
    stack_patch = np.dstack((input_patch, quality_patch))
    return stack_patch


if __name__ == "__main__":
    name = "./sample_image/Bol4_1280x720.jpg"
    test_img = cv2.imread(name)
    cv2.imshow("gaussian_noise", gaussian_noise(test_img).astype('uint8'))
    cv2.imshow("gaussian_noise_y", gaussian_noise_y(test_img, 10).astype('uint8'))
    scikit_gaussian = skimage.util.random_noise(test_img, mode='gaussian', clip=True, var=0.001) * 255
    cv2.imshow("scikit-gaussian_noise", scikit_gaussian.astype('uint8'))
    cv2.imshow("salt_and_pepper_noise", salt_and_pepper_noise(test_img).astype('uint8'))
    cv2.imshow("poisson_noise (not easily visible)", poisson_noise(test_img).astype('uint8'))
    cv2.imshow("speckle_noise", speckle_noise(test_img).astype('uint8'))
    cv2.waitKey(0)
