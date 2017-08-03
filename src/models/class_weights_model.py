from keras.applications import VGG16
from keras.engine import Model

from keras import backend as K, Input
from keras import optimizers
from keras.layers import Conv2D, Lambda, Dense, concatenate, UpSampling2D, Activation

import tensorflow as tf

from src.image_colorization.test import color_images_part

input_shape = (32, 32, 1)
num_classes = 400

weights = "data/weights/hist5.h5"
color_fun = color_images_part


def model():
    """
    Function build and returns model for approach classification with weights

    Returns
    -------
    keras.engine.training.Model
        Keras model for approach
    """

    main_input = Input(shape=input_shape, name='image_part_input')

    x = Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(main_input)
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)

    x = Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
    x = Conv2D(256, (3, 3), padding="same", activation="relu")(x)

    x = Conv2D(256, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
    x = Conv2D(512, (3, 3), padding="same", activation="relu")(x)

    x = Conv2D(512, (3, 3), padding="same", activation="relu")(x)
    main_output = Conv2D(256, (3, 3), padding="same", activation="relu")(x)

    # VGG
    vgg16 = VGG16(weights="imagenet", include_top=True)
    vgg_output = Dense(256, activation='softmax', name='predictions')(vgg16.layers[-2].output)

    def repeat_output(input):
        shape = K.shape(x)
        return K.reshape(K.repeat(input, 4 * 4), (shape[0], 4, 4, 256))

    vgg_output = Lambda(repeat_output)(vgg_output)

    # freeze vgg16
    for layer in vgg16.layers:
        layer.trainable = False

    # concatenated net
    merged = concatenate([vgg_output, main_output], axis=3)

    last = Conv2D(256, (3, 3), padding="same")(merged)

    last = UpSampling2D(size=(2, 2))(last)
    last = Conv2D(256, (3, 3), padding="same", activation="relu")(last)
    last = Conv2D(256, (3, 3), padding="same", activation="relu")(last)

    last = UpSampling2D(size=(2, 2))(last)
    last = Conv2D(256, (3, 3), padding="same", activation="relu")(last)
    last = Conv2D(400, (3, 3), padding="same", activation="relu")(last)

    def resize_image(x):
        return K.resize_images(x, 2, 2, "channels_last")

    # multidimensional softmax
    def custom_softmax(x):
        sh = K.shape(x)
        x = K.reshape(x, (sh[0] * sh[1] * sh[2], num_classes))
        x = K.softmax(x)

        xc = tf.fill(tf.stack([sh[0] * 16 * 16, 1]), 0.0)
        x = K.concatenate([x, xc], axis=-1)

        x = K.reshape(x, (sh[0], sh[1], sh[2], num_classes + 1))
        return x

    last = Activation(custom_softmax)(last)
    last = Lambda(resize_image)(last)

    def categorical_crossentropy_color(y_true, y_pred):
        # Flatten
        shape = K.shape(y_pred)
        y_pred = K.reshape(y_pred, (shape[0] * shape[1] * shape[2], shape[3]))
        y_true = K.reshape(y_true, (shape[0] * shape[1] * shape[2], shape[3]))

        weights = y_true[:, 400:]  # extract weight from y_true
        weights = K.concatenate([weights] * 400, axis=1)
        y_true = y_true[:, :-1]  # remove last column
        y_pred = y_pred[:, :-1]  # remove last column

        # multiply y_true by weights
        y_true = y_true * weights

        cross_ent = K.categorical_crossentropy(y_pred, y_true)
        cross_ent = K.mean(cross_ent, axis=-1)

        return cross_ent

    model = Model(inputs=[main_input, vgg16.input], output=last)
    opt = optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=opt, loss=categorical_crossentropy_color)

    model.summary()
    model.name = "class_with_weights"

    return model
