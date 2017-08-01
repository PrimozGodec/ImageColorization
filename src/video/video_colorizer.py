import inspect
import os

import math
import skvideo.io
import numpy as np
from skimage import color

from src.models import reg_full_model
from src.utils.image_utils import resize_image_lab

source_dir = "../../data/videos/original"
destination_dir = "../../data/videos/colorized"


def get_abs_path(relative):
    """
    Function returns absolute path to the destination.
    It makes paths independent from places where called.

    Parameters
    ----------
    relative : str
        Relative path to the destination dependent on this script

    Returns
    -------
    str
        Absolute path to the destination
    """
    script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # script directory
    return os.path.join(script_dir, relative)


def get_video_list(dir_path):
    """
    This function returns list of videos in directory.
    If any file that you want to color has different ending add it to the list.
    Different files may not be supported for later colorization.

    Parameters
    ----------
    dir_path : str
        Path to the directory

    Returns
    -------
    list of str
        List of file names that contains videos
    """
    image_list = os.listdir(dir_path)
    ext = [".avi", ".mp4"]
    return [im for im in image_list if im.endswith(tuple(ext))]


def color_videos(model):
    # find videos
    videos = get_video_list(get_abs_path(source_dir))
    for video in videos:
        color_one_video(model, video)


def color_one_video(model, video, b_size=32):

    # for each batch
    # metadata
    metadata = skvideo.io.ffprobe(os.path.join(source_dir, video))["video"]
    num_frames = metadata["nb_frames"]
    w, h = metadata["width"], metadata["height"]

    # open reader
    videogen = skvideo.io.vreader(os.path.join(source_dir, video))
    videowriter = skvideo.io.FFmpegWriter(os.path.join(destination_dir, video), (num_frames, w, h, 3))

    for batch_n in range(int(math.ceil(num_frames / b_size))):
        _b_size = b_size if batch_n * b_size <= num_frames else num_frames % b_size

        # load images
        original_size_images = []
        all_images_l = np.zeros((_b_size, 224, 224, 1))
        for i in range(_b_size):
            # get image
            image_rgb = next(videogen)
            image_lab = color.rgb2lab(image_rgb)
            original_size_images.append(image_lab[:, :, 0])
            image_lab_resized = resize_image_lab(image_lab, (224, 224))
            all_images_l[i, :, :, :] = image_lab_resized[:, :, 0][:, :, np.newaxis]

        # prepare images for a global network
        all_vgg = np.zeros((_b_size, 224, 224, 3))
        for i in range(_b_size):
            all_vgg[i, :, :, :] = np.tile(all_images_l[i], (1, 1, 1, 3))

        # color
        if model.name == "reg_full_vgg":  # vgg has no global network
            color_im = model.predict(all_vgg, batch_size=b_size)
        else:
            color_im = model.predict([all_images_l, all_vgg], batch_size=b_size)

        # save all images
        for i in range(_b_size):
            # to rgb
            original_im_bw = original_size_images[i]

            # workaround for not suitable shape while resizing
            small_images = np.concatenate((all_images_l[i], color_im[i]), axis=2)
            colored_im = resize_image_lab(small_images, (h, w))

            lab_im = np.concatenate((original_im_bw[:, :, np.newaxis], colored_im[:, :, 1:]), axis=2)
            im_rgb = color.lab2rgb(lab_im)

            # save
            videowriter.writeFrame(im_rgb)
        print(batch_n)
    videowriter.close()

if __name__ == "__main__":

    # load model
    model = reg_full_model.model()

    # load weights
    model.load_weights(reg_full_model.weights)

    color_videos(model)

    print("done")
