import os
import urllib.request

import scipy.misc
from PIL import Image
import numpy as np
from progressbar import ProgressBar, Percentage, Bar
from skimage import color


def load_images(file):
    """
    Function loads black and white image and convert it to CIE LAB

    Parameters
    ----------
    file : str
        Path to image file

    Returns
    -------
    ndarray
        3D array with image
    """

    try:
        img = Image.open(file)
    except (OSError, ValueError, IOError, ZeroDivisionError) as e:
        print("Can not open file", file, "Error: ", e)
        return None
    img = img.convert(mode="RGB")  # ensure that image rgb
    rgb = np.array(img)

    return color.rgb2lab(rgb)


def resize_image_lab(im, size):
    """
    This function resizes images of any colorspace and number of channels

    Parameters
    ----------
    im : ndarray
        Numpy array that contains image
    size : (int, int)
        New size of image

    Returns
    -------
    ndarray
        Resized image
    """
    # because there is not good resizer for a LAB images
    im = (color.lab2rgb(im) * 255).astype(int)
    img = scipy.misc.imresize(im, size)
    return color.rgb2lab(np.array(img))


def get_weights(file_name):
    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/weights")
    weights_url = "https://github.com/PrimozGodec/ImageColorization/releases/download/v0.0.1"


    # if file do not exist download it
    print(os.path.join(weights_dir, file_name))
    if not os.path.isfile(os.path.join(weights_dir, file_name)):
        # init progress bar
        pbar = ProgressBar(widgets=[Percentage(), Bar()])

        def show_progress(count, block_size, total_size):
            pbar.update(int(count * block_size * 100 / total_size))

        # download
        urllib.request.urlretrieve(os.path.join(weights_url, file_name),
                                   os.path.join(weights_dir, file_name),
                                   reporthook=show_progress)

    return file_name
