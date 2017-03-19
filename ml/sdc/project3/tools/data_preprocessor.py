import numpy as np
import cv2
from sklearn import preprocessing

from parameters import *


class DataPreprocessor:

    def preprocess(self, X, y):
        X = np.array(X).astype('float32')
        y = np.array(y).astype('float32')

        # X = self.__crop(X)
        # X = self.__resize(X)
        # X = self.__normalize(X)

        # y = self.steering_angle_to_label(y)

        return X, y

    def preprocess_to_drive(self, X):
        # X = np.array(X).astype('float32')

        # X = self.__crop(X)
        # X = self.__resize(X)
        # X = self.__normalize(X)

        return X

    def label_to_one_hot_encoding(self, y):
        lb = preprocessing.LabelBinarizer()
        lb.fit(list(range(-24, 25)))
        return lb.transform(y)

    def steering_angle_to_label(self, y):
        y = y * 100 / 4
        y = y.astype('int')
        return y

    def label_to_steering_angle(self, label):
        return (label - 24) / 100. * 4

    def __normalize(self, X):
        X /= 255
        return X

    def __crop(self, X):

        cropped = []
        for image in X:
            height = image.shape[0]

            crop_top = np.math.floor(height * 0.2)
            crop_bottom = height - np.math.floor(height * 0.20)

            result = image[crop_top: crop_bottom, :]
            result = np.array(result)

            cropped.append(result)

        return cropped

    def __resize(self, X):

        resized = []
        for image in X:
            res = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
            res = np.array(res)
            resized.append(res)

        return np.array(resized)


