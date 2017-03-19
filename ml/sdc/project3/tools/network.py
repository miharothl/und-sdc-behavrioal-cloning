from keras.layers import Convolution2D, Lambda, Cropping2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from parameters import *
from keras.regularizers import l2


class Network():
    def __init__(self):
        pass

    def create_convolutional_nvidia_style_modified(self, input_shape, num_classes):

        model = Sequential()

        model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
        model.add(Cropping2D(cropping=((70, 25), (0, 0))))

        model.add(Convolution2D(24, 5, 6, border_mode='same', subsample=(2, 4), ))
        model.add(Activation('relu'))
        model.add(Dropout(DROPOUT))

        model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
        model.add(Activation('relu'))
        model.add(Dropout(DROPOUT))

        model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
        model.add(Activation('relu'))
        model.add(Dropout(DROPOUT))

        model.add(Convolution2D(64, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Dropout(DROPOUT))

        model.add(Convolution2D(64, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Dropout(DROPOUT))

        model.add(Flatten())

        model.add(Dense(300, W_regularizer=l2(L2_REGULARIZATION)))
        model.add(Activation('relu'))
        model.add(Dropout(DROPOUT))

        model.add(Dense(75))
        model.add(Activation('relu'))
        model.add(Dropout(DROPOUT))

        model.add(Dense(40))
        model.add(Activation('relu'))
        model.add(Dropout(DROPOUT))

        model.add(Dense(10))
        model.add(Activation('relu'))
        model.add(Dropout(DROPOUT))

        model.add(Dense(1))

        return model

    def create_convolutional_nvidia_style(self, input_shape, num_classes):

        model = Sequential()

        model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2),
                                input_shape=input_shape))
        model.add(Activation('relu'))

        model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
        model.add(Activation('relu'))

        model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
        model.add(Activation('relu'))

        model.add(Convolution2D(64, 3, 3, border_mode='same'))
        model.add(Activation('relu'))

        model.add(Convolution2D(64, 3, 3, border_mode='same'))
        model.add(Activation('relu'))

        model.add(Flatten())

        model.add(Dense(1164))
        model.add(Activation('relu'))
        model.add(Dropout(DROPOUT))

        model.add(Dense(100))
        model.add(Activation('relu'))

        model.add(Dense(50))
        model.add(Activation('relu'))

        model.add(Dense(10))
        model.add(Activation('relu'))
        model.add(Dropout(DROPOUT))

        model.add(Dense(1))

        return model

    def create_simple(self, input_shape, num_classes):
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(1))
        return model
