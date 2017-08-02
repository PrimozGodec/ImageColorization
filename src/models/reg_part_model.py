from keras.applications import VGG16
from keras.engine import Model

from keras import backend as K, Input
from keras import optimizers
from keras.layers import Conv2D, Lambda, Dense, concatenate, regularizers, add, Conv2DTranspose, MaxPooling2D

from src.image_colorization.test import color_images_part

input_shape = (32, 32, 1)

weights = "data/weights/imp9.h5"
color_fun = color_images_part


def model():
    """
    Function build and returns model for approach regression part

    Returns
    -------
    keras.engine.training.Model
        Keras model for approach
    """
    main_input = Input(shape=input_shape, name='image_part_input')

    x = Conv2D(64, (3, 3), padding="same", activation="relu",
               kernel_regularizer=regularizers.l2(0.01))(main_input)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x1 = Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(x1)
    x = Conv2D(128, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = add([x, x1])

    x = Conv2D(128, (3, 3), padding="same", activation="relu",
               kernel_regularizer=regularizers.l2(0.01))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x1 = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(x1)
    x = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = add([x, x1])

    x = Conv2D(256, (3, 3), padding="same", activation="relu",
               kernel_regularizer=regularizers.l2(0.01))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x1 = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(x1)
    x = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = add([x, x1])

    x = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    main_output = Conv2D(256, (3, 3), padding="same", activation="relu",
                         kernel_regularizer=regularizers.l2(0.01))(x)

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

    last = Conv2D(128, (3, 3), padding="same")(merged)

    last = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="relu",
                           kernel_regularizer=regularizers.l2(0.01))(last)
    last = Conv2D(64, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(last)
    last = Conv2D(64, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(last)

    last = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="relu",
                           kernel_regularizer=regularizers.l2(0.01))(last)
    last = Conv2D(32, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(last)
    last = Conv2D(2, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(last)

    def resize_image(x):
        return K.resize_images(x, 2, 2, "channels_last")

    def unormalise(x):
        # outputs in range [0, 1] resized to range [-100, 100]
        return (x * 200) - 100

    last = Lambda(resize_image)(last)
    last = Lambda(unormalise)(last)

    def custom_mse(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=[1, 2, 3])

    model = Model(inputs=[main_input, vgg16.input], output=last)
    opt = optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=opt, loss=custom_mse)

    model.summary()
    model.name = "reg_part"

    return model
