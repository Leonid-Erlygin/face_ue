import numpy as np
from scipy import misc
from scipy.ndimage.filters import gaussian_filter
import skimage
from skimage.transform import resize as skimage_resize
import imageio


def noisy(image, noise_type):
    if noise_type == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy


def add_noise(images, noise_type="gauss"):
    images_new = images.copy()
    for i in range(images_new.shape[0]):
        images_new[i] = noisy(images[i], noise_type)

    return images_new


def gaussian_blur(images, sigma=5):
    images_new = images.copy()
    for i in range(images_new.shape[0]):
        images_new[i] = gaussian_filter(images_new[i], sigma=sigma)

    return images_new


def get_new_shape(images, size=None, n=None):
    shape = list(images.shape)
    if size is not None:
        h, w = tuple(size)
        shape[1] = h
        shape[2] = w
    if n is not None:
        shape[0] = n
    shape = tuple(shape)
    return shape


def random_crop(images, size):
    n, _h, _w = images.shape[:3]
    h, w = tuple(size)
    shape_new = get_new_shape(images, size)
    assert _h >= h and _w >= w

    images_new = np.ndarray(shape_new, dtype=images.dtype)

    y = np.random.randint(low=0, high=_h - h + 1, size=(n))
    x = np.random.randint(low=0, high=_w - w + 1, size=(n))

    for i in range(n):
        images_new[i] = images[i, y[i] : y[i] + h, x[i] : x[i] + w]

    return images_new


def center_crop(images, size):
    n, _h, _w = images.shape[:3]
    h, w = tuple(size)
    assert _h >= h and _w >= w

    y = int(round(0.5 * (_h - h)))
    x = int(round(0.5 * (_w - w)))

    images_new = images[:, y : y + h, x : x + w]

    return images_new


def random_flip(images):
    images_new = images.copy()
    flips = np.random.rand(images_new.shape[0]) >= 0.5

    for i in range(images_new.shape[0]):
        if flips[i]:
            images_new[i] = np.fliplr(images[i])

    return images_new


def flip(images):
    images_new = images.copy()
    for i in range(images_new.shape[0]):
        images_new[i] = np.fliplr(images[i])

    return images_new


def resize(images, size):
    n, _h, _w = images.shape[:3]
    h, w = tuple(size)
    shape_new = get_new_shape(images, size)

    images_new = np.ndarray(shape_new, dtype=images.dtype)

    for i in range(n):
        # images_new[i] = skimage.transform.resize(images[i], (h, w))
        images_new[i] = skimage_resize(images[i], (h, w))

    return images_new


def padding(images, padding):
    n, _h, _w = images.shape[:3]
    if len(padding) == 2:
        pad_t = pad_b = padding[0]
        pad_l = pad_r = padding[1]
    else:
        pad_t, pad_b, pad_l, pad_r = tuple(padding)

    size_new = (_h + pad_t + pad_b, _w + pad_l + pad_b)
    shape_new = get_new_shape(images, size_new)
    images_new = np.zeros(shape_new, dtype=images.dtype)
    images_new[:, pad_t : pad_t + _h, pad_l : pad_l + _w] = images

    return images_new


def standardize_images(images, standard):
    if standard == "mean_scale":
        mean = 127.5
        std = 128.0
    elif standard == "scale":
        mean = 0.0
        std = 255.0
    else:
        raise RuntimeError("Don't know this type")
    images_new = images.astype(np.float32)
    images_new = (images_new - mean) / std
    return images_new


def random_shift(images, max_ratio):
    n, _h, _w = images.shape[:3]
    pad_x = int(_w * max_ratio) + 1
    pad_y = int(_h * max_ratio) + 1
    images_temp = padding(images, (pad_x, pad_y))
    images_new = images.copy()

    shift_x = (_w * max_ratio * np.random.rand(n)).astype(np.int32)
    shift_y = (_h * max_ratio * np.random.rand(n)).astype(np.int32)

    for i in range(n):
        images_new[i] = images_temp[
            i,
            pad_y + shift_y[i] : pad_y + shift_y[i] + _h,
            pad_x + shift_x[i] : pad_x + shift_x[i] + _w,
        ]

    return images_new


def random_downsample(images, min_ratio):
    n, _h, _w = images.shape[:3]
    images_new = images.copy()
    ratios = min_ratio + (1 - min_ratio) * np.random.rand(n)

    for i in range(n):
        w = int(round(ratios[i] * _w))
        h = int(round(ratios[i] * _h))
        images_new[i, :h, :w] = misc.imresize(images[i], (h, w))
        images_new[i] = misc.imresize(images_new[i, :h, :w], (_h, _w))

    return images_new


def random_interpolate(images):
    _n, _h, _w = images.shape[:3]
    nd = images.ndim - 1
    assert _n % 2 == 0
    n = int(_n / 2)

    ratios = np.random.rand(n, *([1] * nd))
    images_left, images_right = (images[np.arange(n) * 2], images[np.arange(n) * 2 + 1])
    images_new = ratios * images_left + (1 - ratios) * images_right
    images_new = images_new.astype(np.uint8)

    return images_new


def expand_flip(images):
    """Flip each image in the array and insert it after the original image."""
    _n, _h, _w = images.shape[:3]
    shape_new = get_new_shape(images, n=2 * _n)
    images_new = np.stack([images, flip(images)], axis=1)
    images_new = images_new.reshape(shape_new)
    return images_new


def five_crop(images, size):
    _n, _h, _w = images.shape[:3]
    h, w = tuple(size)
    assert h <= _h and w <= _w

    shape_new = get_new_shape(images, size, n=5 * _n)
    images_new = []
    images_new.append(images[:, :h, :w])
    images_new.append(images[:, :h, -w:])
    images_new.append(images[:, -h:, :w])
    images_new.append(images[:, -h:, -w:])
    images_new.append(center_crop(images, size))
    images_new = np.stack(images_new, axis=1).reshape(shape_new)
    return images_new


def ten_crop(images, size):
    _n, _h, _w = images.shape[:3]
    shape_new = get_new_shape(images, size, n=10 * _n)
    images_ = five_crop(images, size)
    images_flip_ = five_crop(flip(images), size)
    images_new = np.stack([images_, images_flip_], axis=1)
    images_new = images_new.reshape(shape_new)
    return images_new


register = {
    "resize": resize,
    "padding": padding,
    "random_crop": random_crop,
    "center_crop": center_crop,
    "random_flip": random_flip,
    "standardize": standardize_images,
    "random_shift": random_shift,
    "random_interpolate": random_interpolate,
    "random_downsample": random_downsample,
    "expand_flip": expand_flip,
    "five_crop": five_crop,
    "ten_crop": ten_crop,
    "flip": flip,
    "add_noise": add_noise,
    "gaussian_blur": gaussian_blur,
}


def preprocess(
    images, center_crop_size, mode="RGB", align: tuple = None, *, is_training=False
):
    """
    #TODO: docs, describe mode parameter
    """
    # TODO: this is not preprocess actually
    if type(images[0]) == str:
        image_paths = images
        images = []
        for image_path in image_paths:
            images.append(imageio.imread(image_path, pilmode=mode))
        images = np.stack(images, axis=0)
    else:
        assert type(images) == np.ndarray
        assert images.ndim == 4

    # Process images
    preprocess_train = [
        ["center_crop", center_crop_size],
        ["random_flip"],
        # ["gaussian_blur", 0.5],
        # ["add_noise"],
        ["standardize", "mean_scale"],
    ]
    preprocess_test = [
        ["center_crop", center_crop_size],
        ["standardize", "mean_scale"],
    ]
    proc_funcs = preprocess_train if is_training else preprocess_test

    for proc in proc_funcs:
        proc_name, proc_args = proc[0], proc[1:]
        assert (
            proc_name in register
        ), "Not a registered preprocessing function: {}".format(proc_name)
        images = register[proc_name](images, *proc_args)
    if len(images.shape) == 3:
        images = images[:, :, :, None]
    return images


# (horizontal_flip, center_cropp/random_cropp, gausian_blur/median_blur, noise? , hue/saturation?)
def preprocess_tta(
    images, center_crop_size, mode="RGB", align: tuple = None, *, is_training=False
):
    """
    #TODO: docs, describe mode parameter
    """
    # TODO: this is not preprocess actually
    if type(images[0]) == str:
        image_paths = images
        images = []
        for image_path in image_paths:
            images.append(imageio.imread(image_path, pilmode=mode))
        images = np.stack(images, axis=0)
    else:
        assert type(images) == np.ndarray
        assert images.ndim == 4

    preprocess_augmentation_orig = [
        ["center_crop", center_crop_size],
        ["standardize", "mean_scale"],
    ]
    preprocess_augmentation_flipped = [
        ["center_crop", center_crop_size],
        ["flip"],
        ["standardize", "mean_scale"],
    ]
    preprocess_augmentation_blurred = [
        ["center_crop", center_crop_size],
        ["gaussian_blur"],
        ["standardize", "mean_scale"],
    ]
    preprocess_augmentation_noise = [
        ["center_crop", center_crop_size],
        ["add_noise"],
        ["standardize", "mean_scale"],
    ]

    proc_funcs_list = [
        preprocess_augmentation_orig,
        preprocess_augmentation_flipped,
        preprocess_augmentation_blurred,
        preprocess_augmentation_noise,
    ]

    images_augmented = []

    for proc_funcs in proc_funcs_list:
        for proc in proc_funcs:
            proc_name, proc_args = proc[0], proc[1:]
            assert (
                proc_name in register
            ), "Not a registered preprocessing function: {}".format(proc_name)
            images = register[proc_name](images, *proc_args)
        if len(images.shape) == 3:
            images = images[:, :, :, None]

        images_augmented.append(images)

    return images_augmented


def preprocess_gan(
    images, resize_size, mode="RGB", align: tuple = None, *, is_training=False
):
    """
    #TODO: docs, describe mode parameter
    """
    # TODO: this is not preprocess actually
    if type(images[0]) == str:
        image_paths = images
        images = []
        for image_path in image_paths:
            images.append(imageio.imread(image_path, pilmode=mode))
        images = np.stack(images, axis=0)
    else:
        assert type(images) == np.ndarray
        assert images.ndim == 4

    # Process images
    preprocess_train = [
        ["standardize", "mean_scale"],
        ["resize", resize_size],
        ["random_flip"],
    ]
    preprocess_test = [
        ["standardize", "mean_scale"],
        ["resize", resize_size],
    ]
    proc_funcs = preprocess_train if is_training else preprocess_test

    for proc in proc_funcs:
        proc_name, proc_args = proc[0], proc[1:]
        assert (
            proc_name in register
        ), "Not a registered preprocessing function: {}".format(proc_name)
        images = register[proc_name](images, *proc_args)

    if len(images.shape) == 3:
        images = images[:, :, :, None]
    return images


def preprocess_magface(
    images, center_crop_size, mode="RGB", align: tuple = None, *, is_training=False
):
    """
    #TODO: docs, describe mode parameter
    """
    # TODO: this is not preprocess actually
    if type(images[0]) == str:
        image_paths = images
        images = []
        for image_path in image_paths:
            images.append(imageio.imread(image_path, pilmode=mode))
        images = np.stack(images, axis=0)
    else:
        assert type(images) == np.ndarray
        assert images.ndim == 4

    # Process images
    preprocess_train = [
        ["center_crop", center_crop_size],
        ["random_flip"],
        ["standardize", "mean_scale"],
    ]
    preprocess_test = [
        ["center_crop", center_crop_size],
        ["standardize", "scale"],
    ]
    proc_funcs = preprocess_train if is_training else preprocess_test

    for proc in proc_funcs:
        proc_name, proc_args = proc[0], proc[1:]
        assert (
            proc_name in register
        ), "Not a registered preprocessing function: {}".format(proc_name)
        images = register[proc_name](images, *proc_args)
    if len(images.shape) == 3:
        images = images[:, :, :, None]
    return images


def preprocess_blurred(
    images, center_crop_size, mode="RGB", blur_intensity=1, is_training=False
):
    """
    #TODO: docs, describe mode parameter
    """
    # TODO: this is not preprocess actually
    if type(images[0]) == str:
        image_paths = images
        images = []
        for image_path in image_paths:
            images.append(imageio.imread(image_path, pilmode=mode))
        images = np.stack(images, axis=0)
    else:
        assert type(images) == np.ndarray
        assert images.ndim == 4

    # Process images
    preprocess_train = [
        ["center_crop", center_crop_size],
        ["random_flip"],
        ["gaussian_blur", blur_intensity],
        ["standardize", "mean_scale"],
    ]
    preprocess_test = [
        ["center_crop", center_crop_size],
        ["gaussian_blur", blur_intensity],
        ["standardize", "mean_scale"],
    ]
    proc_funcs = preprocess_train if is_training else preprocess_test

    for proc in proc_funcs:
        proc_name, proc_args = proc[0], proc[1:]
        assert (
            proc_name in register
        ), "Not a registered preprocessing function: {}".format(proc_name)
        images = register[proc_name](images, *proc_args)
    if len(images.shape) == 3:
        images = images[:, :, :, None]
    return images
