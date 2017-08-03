import inspect
import os

import math
import skvideo.io
import numpy as np
from progressbar import ProgressBar, Percentage, Bar, ETA
from skimage import color

from src.models import reg_full_model
from src.utils.image_utils import resize_image_lab

source_dir = "../../data/video/original"
destination_dir = "../../data/video/colorized"


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
    """
    Function finds all videos to colorize and call colorization on each

    Parameters
    ----------
    model : keras.engine.training.Model
        Model used for colorization
    """
    # find videos
    videos = get_video_list(get_abs_path(source_dir))
    for video in videos:
        color_one_video(model, video)


def color_one_video(model, video, b_size=32):
    """
    Function color one video_colorization and save it to destination directory

    Parameters
    ----------
    model : keras.engine.training.Model
        Model used fro colorization
    video : str
        Name of video_colorization to color. Video is situated in source directory
    b_size : int
        Size of frames that are colored in one step
    """
    # metadata
    metadata = skvideo.io.ffprobe(os.path.join(get_abs_path(source_dir), video))["video"]
    num_frames = int(metadata["@nb_frames"])
    w, h = int(metadata["@width"]), int(metadata["@height"])
    frame_rate = metadata["@r_frame_rate"].split("/")
    frame_rate = str(float(frame_rate[0]) / float(frame_rate[1]))

    print(skvideo.io.ffprobe(os.path.join(get_abs_path(source_dir), video))["audio"])

    # open reader and writer
    videogen = skvideo.io.vreader(os.path.join(get_abs_path(source_dir), video))
    videowriter = skvideo.io.FFmpegWriter(os.path.join(get_abs_path(destination_dir), video),
                                          inputdict={"-r": frame_rate},
                                          outputdict={"-r": frame_rate})

    # progress bar
    print("Starting", video)
    pbar = ProgressBar(maxval=num_frames, widgets=[Percentage(), ' ', Bar(), ' ', ETA()])
    pbar.start()

    # for each batch
    for batch_n in range(int(math.ceil(num_frames / b_size))):
        _b_size = b_size if (batch_n + 1) * b_size <= num_frames else num_frames % b_size

        # load images
        original_size_images = []
        all_images_l = np.zeros((_b_size, 224, 224, 1))
        for i in range(_b_size):
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
            im_rgb = (color.lab2rgb(lab_im) * 255).astype(int)

            # save
            videowriter.writeFrame(im_rgb)

        # update progress bar
        pbar.update(min((batch_n + 1) * b_size, num_frames))

    # end with progress bar
    pbar.finish()

    videogen.close()
    videowriter.close()

if __name__ == "__main__":

    # load model
    model = reg_full_model.model()

    # load weights
    model.load_weights(reg_full_model.weights)

    color_videos(model)

    print("done")
