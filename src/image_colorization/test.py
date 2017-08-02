import inspect
import os

import numpy as np
import scipy.misc
from progressbar import ProgressBar, Percentage, Bar, ETA
from skimage import color
import math

from src.utils.image_utils import load_images, resize_image_lab


# variables
data_origin = "../../data/image/original"
data_destination = "../../data/image/colorized/"


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


def get_image_list(dir_path):
    """
    This function returns list of images in directory.
    If any file that you want to color has different ending add it to the list.
    Different files may not be supported for later colorization.

    Parameters
    ----------
    dir_path : str
        Path to the directory

    Returns
    -------
    list of str
        List of file names that contains images
    """
    image_list = os.listdir(dir_path)
    ext = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png"]
    return [im for im in image_list if im.endswith(tuple(ext))]


def color_images_full(model, b_size=32):
    """
    Function that colors images with approaches on full images.
    Function is used on reg-full and reg-full-vgg approaches.

    Parameters
    ----------
    model : keras.engine.training.Model
        Model for image colorization
    b_size : int
        Size of bach of images
    """
    abs_file_path = get_abs_path(data_origin)
    images = get_image_list(abs_file_path)

    # get list of images to color
    num_of_images = len(images)

    #progress bar
    pbar = ProgressBar(maxval=num_of_images, widgets=[Percentage(), ' ', Bar(), ' ', ETA()])
    pbar.start()

    # for each batch
    for batch_n in range(int(math.ceil(num_of_images / b_size))):
        _b_size = b_size if (batch_n + 1) * b_size < num_of_images else num_of_images % b_size

        # load images
        original_size_images = []
        all_images_l = np.zeros((_b_size, 224, 224, 1))
        for i in range(_b_size):
            # get image
            image_lab = load_images(os.path.join(abs_file_path, images[batch_n * b_size + i]))
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
        abs_save_path = get_abs_path(data_destination)
        for i in range(_b_size):
            # to rgb
            original_im_bw = original_size_images[i]
            h, w = original_im_bw.shape

            # workaround for not suitable shape while resizing
            small_images = np.concatenate((all_images_l[i], color_im[i]), axis=2)
            colored_im = resize_image_lab(small_images, (h, w))

            lab_im = np.concatenate((original_im_bw[:, :, np.newaxis], colored_im[:, :, 1:]), axis=2)
            im_rgb = color.lab2rgb(lab_im)

            # save
            scipy.misc.toimage(im_rgb, cmin=0.0, cmax=1.0).save(
                abs_save_path + model.name + "_" + images[batch_n * b_size + i])

        # update progress bar
        pbar.update(min((batch_n + 1) * b_size, num_of_images))

    # stop progress bar
    pbar.finish()


# matrices for multiplying that needs to calculate only once
# matrices are used in coloring on parts
vec = np.hstack((np.linspace(1/16, 1 - 1/16, 16), np.flip(np.linspace(1/16, 1 - 1/16, 16), axis=0)))
one = np.ones((32, 32))
xv, yv = np.meshgrid(vec, vec)
weight_m = xv * yv
weight_left = np.hstack((one[:, :16], xv[:, 16:])) * yv
weight_right = np.hstack((xv[:, :16], one[:, 16:])) * yv
weight_top = np.vstack((one[:16, :], yv[16:, :])) * xv
weight_bottom = np.vstack((yv[:16, :], one[16:, :])) * xv

weight_top_left = np.hstack((one[:, :16], xv[:, 16:])) * np.vstack((one[:16, :], yv[16:, :]))
weight_top_right = np.hstack((xv[:, :16], one[:, 16:])) * np.vstack((one[:16, :], yv[16:, :]))
weight_bottom_left = np.hstack((one[:, :16], xv[:, 16:])) * np.vstack((yv[:16, :], one[16:, :]))
weight_bottom_right = np.hstack((xv[:, :16], one[:, 16:])) * np.vstack((yv[:16, :], one[16:, :]))


def color_images_part(model):
    """
    Function that colors images with approaches on part of images.
    Function is used on reg-parts, class-wo-weights and class-with-weights approaches.

    Parameters
    ----------
    model : keras.engine.training.Model
        Model for image colorization
    """

    # get images to color
    test_set_dir_path = get_abs_path(data_origin)

    image_list = get_image_list(test_set_dir_path)
    num_of_images = len(image_list)

    # init progress bar
    pbar = ProgressBar(maxval=num_of_images, widgets=[Percentage(), ' ', Bar(), ' ', ETA()])
    pbar.start()

    # repeat for each image
    for i in range(num_of_images):
        # get image
        image_lab = load_images(os.path.join(test_set_dir_path, image_list[i]))
        image_l = image_lab[:, :, 0]
        h, w = image_l.shape

        # split images to list of images
        slices_dim_h = int(math.ceil(h/32))
        slices_dim_w = int(math.ceil(w/32))

        slices = np.zeros((slices_dim_h * slices_dim_w * 4, 32, 32, 1))
        for a in range(slices_dim_h * 2 - 1):
            for b in range(slices_dim_w * 2 - 1):
                part = image_l[a*32//2: a*32//2 + 32, b*32//2: b*32//2 + 32]
                # fill with zero on edges
                _part = np.zeros((32, 32))
                _part[:part.shape[0], :part.shape[1]] = part

                slices[a * slices_dim_w * 2 + b] = _part[:, :, np.newaxis]

        # lover originals dimension to 224x224 to feed vgg and increase dim
        image_lab_224_b = resize_image_lab(image_lab, (224, 224))
        image_l_224 = np.repeat(image_lab_224_b[:, :, 0, np.newaxis], 3, axis=2).astype(float)

        # append together booth lists
        input_data = [slices, np.array([image_l_224, ] * slices_dim_h * slices_dim_w * 4)]

        # predict
        predictions_ab = model.predict(input_data, batch_size=32)

        # for histograms -> transformation from hist to ab
        if model.name == "class_wo_weights" or model.name == "class_with_weights":
            indices = np.argmax(predictions_ab[:, :, :, :], axis=3)

            predictions_a = indices // 20 * 10 - 100 + 5
            predictions_b = indices % 20 * 10 - 100 + 5  # +5 to set in the middle box
            predictions_ab = np.stack((predictions_a, predictions_b), axis=3)

        # reshape back to original size
        original_size_im = np.zeros((slices_dim_h * 32, slices_dim_w * 32, 2))
        o_h, o_w = original_size_im.shape[:2]

        for n in range(predictions_ab.shape[0]):
            a, b = n // (slices_dim_w * 2) * 16, n % (slices_dim_w * 2) * 16

            if a + 32 > o_h or b + 32 > o_w:
                continue  # it is empty edge

            # weight decision
            if a == 0 and b == 0:
                weight = weight_top_left
            elif a == 0 and b == o_w - 32:
                weight = weight_top_right
            elif a == 0:
                weight = weight_top
            elif a == o_h - 32 and b == 0:
                weight = weight_bottom_left
            elif b == 0:
                weight = weight_left
            elif a == o_h - 32 and b == o_w - 32:
                weight = weight_bottom_right
            elif a == o_h - 32:
                weight = weight_bottom
            elif b == o_w - 32:
                weight = weight_right
            else:
                weight = weight_m

            im_a = predictions_ab[n, :, :, 0] * weight
            im_b = predictions_ab[n, :, :, 1] * weight

            original_size_im[a:a+32, b:b+32, :] += np.stack((im_a, im_b), axis=2)

        # make original shape image
        original_size_im = original_size_im[:h, :w]

        # to rgb
        color_im = np.concatenate((image_l[:, :, np.newaxis], original_size_im), axis=2)
        im_rgb = color.lab2rgb(color_im)

        # save
        abs_svave_path = get_abs_path(data_destination)
        scipy.misc.toimage(im_rgb, cmin=0.0, cmax=1.0).save(abs_svave_path + model.name + "_" + image_list[i])

        # update progress bar
        pbar.update(i + 1)

    # finish progress bar
    pbar.finish()
