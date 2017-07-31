import inspect
import os

import numpy as np
import scipy.misc
from skimage import color
import math

from src.utils.image_utils import resize_image, load_images


def get_abs_path(relative):
    script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # script directory
    return os.path.join(script_dir, relative)


def get_image_list(dir_path):
    image_list = os.listdir(dir_path)
    ext = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png"]
    return [im for im in image_list if im.endswith(tuple(ext))]


def color_images_full(model, name, b_size=32):
    """
    reg-full
    """

    abs_file_path = get_abs_path("../../data/original")
    images = get_image_list(abs_file_path)

    # get list of images to color
    num_of_images = len(images)

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
            image_lab_resized = resize_image(image_lab, (224, 224), "LAB")
            all_images_l[i, :, :, :] = image_lab_resized[:, :, 0][:, :, np.newaxis]

            print(original_size_images[i, :5, :5])
            print(image_lab_resized[i, :5, :5, :])

        # prepare images for a global network
        all_vgg = np.zeros((_b_size, 224, 224, 3))
        for i in range(_b_size):
            all_vgg[i, :, :, :] = np.tile(all_images_l[i], (1, 1, 1, 3))

        # color
        color_im = model.predict([all_images_l, all_vgg], batch_size=b_size)

        # save all images
        abs_save_path = get_abs_path("../../data/colorized/")
        for i in range(_b_size):
            # to rgb
            original_im_bw = original_size_images[i]
            h, w = original_im_bw.shape

            scipy.misc.toimage(original_im_bw, cmin=0.0, cmax=100.0).save(
                abs_save_path + "test1" + name + images[batch_n * b_size + i])
            scipy.misc.toimage(all_images_l[i, :, :, 0], cmin=0.0, cmax=100.0).save(
                abs_save_path + "test2" + name + images[batch_n * b_size + i])



            # workaround for not suitable shape while resizing
            small_images = np.concatenate((all_images_l[i], color_im[i]), axis=2)

            im_rgb = color.lab2rgb(small_images)
            scipy.misc.toimage(im_rgb, cmin=0.0, cmax=1.0).save(abs_save_path + "test" +name + images[batch_n * b_size + i])

            colored_im = resize_image(small_images, (w, h), "LAB")

            lab_im = np.concatenate((original_im_bw[:, :, np.newaxis], colored_im[:, :, 1:]), axis=2)
            im_rgb = color.lab2rgb(lab_im)

            # save
            scipy.misc.toimage(im_rgb, cmin=0.0, cmax=1.0).save(
                abs_save_path + name + images[batch_n * b_size + i])
