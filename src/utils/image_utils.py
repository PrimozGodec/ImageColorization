import PIL
import scipy.misc
from PIL import Image
import numpy as np
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


def resize_image_lab(im, size, mode):
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
    im = (color.lab2rgb(im) * 255).astype(int)
    print(im)
    img = Image.fromarray(im, "RGB")
    img = img.resize(size, Image.ANTIALIAS)
    print(np.array(img))
    return color.rgb2lab(np.array(img))
