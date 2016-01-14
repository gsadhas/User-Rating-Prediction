from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.models import model_from_json
from keras.regularizers import activity_l1l2, l1l2
from keras.optimizers import SGD

import utility


def get_cnn_kmodel(size, image_dim):
    """
    Generates CNN model using keras
    :return: return generated model
    """
    model = Sequential()
    # add convolution layers
    x = 32

    # for i in range(1):
    #     model.add(Convolution2D(x, 3, 3, border_mode='full', activation='relu',
    #                             input_shape=(image_dim, size, size)))
    #     model.add(Convolution2D(x, 2, 2, border_mode='full', activation='relu'))
    #     model.add(Convolution2D(x, 2, 2, border_mode='full', activation='relu'))
    #
    #     model.add(MaxPooling2D(pool_size=(2, 2)))
    #     x *= 2

    # for i in range(1):
    #     model.add(Convolution2D(x, 3, 3, border_mode='full',
    #                             activation='relu',
    #                             input_shape=(image_dim, size, size)))
    #     model.add(Convolution2D(x, 2, 2,activation='relu'))
    #     model.add(MaxPooling2D(pool_size=(2, 2)))
    #     x *= 2
    #
    # for i in range(3):
    #     model.add(Convolution2D(x, 3, 3, border_mode='valid', activation='relu',
    #                             input_shape=(image_dim, size, size)))
    #     model.add(MaxPooling2D(pool_size=(2, 2)))
    #     x *= 2

    # for i in range(1):
    #     model.add(Convolution1D(x, 3, border_mode='full',
    #                             activation='relu',
    #                             input_shape=(1,size)))
    #     model.add(Convolution1D(x, 2,border_mode='full',activation='relu'))
    #     model.add(Convolution1D(x, 2,border_mode='full',activation='relu'))
    #     model.add(MaxPooling1D(pool_length=2))
    #     x *= 2

    # for i in range(1):
    #     model.add(Convolution1D(x, 3, border_mode='full',
    #                             activation='relu',
    #                             input_shape=(1,size)))
    #     model.add(Convolution1D(x, 2,border_mode='full',activation='relu'))
    #     model.add(MaxPooling1D(pool_length=2))
    #     x *= 2

    for i in range(3):
        model.add(Convolution1D(x, 3, border_mode='full', activation='relu',
                                input_shape=(image_dim, size)))
        model.add(MaxPooling1D(pool_length=2))
        x *= 2

    # start adding dense layers
    model.add(Flatten())

    # final output layer
    model.add(Dense(5, activation='softmax'))

    sgd = SGD(lr=0.1, momentum=0.9, decay=1e-7, nesterov=True)
    # sgd = Adam(lr=0.005, beta_1=0.85, beta_2=0.99, epsilon=1e-5, verbose=1)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    # 'rmsprop'

    return model


def save_cnn_model(model, save_path=None):
    """
    save cnn model to disk
    :param model: cnn model
    :param save_path: path to save model to
    :return:
    """
    if save_path is None:
        arch = "models/cnn_arch.json"
        weights = "data/models/cnn_weights.h5"
    else:
        utility.assure_path_exists(save_path)
        arch = save_path + "cnn_arch.json"
        weights = save_path + "cnn_weights"

    # save architecture
    json_string = model.to_json()
    open(arch, 'w').write(json_string)
    # save model weights
    model.save_weights(weights, overwrite=True)


def load_cnn_model(model_arch="cnn_arch.json",
                   model_weights="cnn_weights"):
    """
    load a cnn model from disk
    :param model_arch: arch json path
    :param model_weights: weights path .h5
    :return: returns model
    """
    model = model_from_json(open(model_arch).read())
    model.load_weights(model_weights)
    return model
