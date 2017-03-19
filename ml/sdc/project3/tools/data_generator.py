import random

import cv2
import numpy as np
from parameters import *


class DataGenerator:
    def __init__(self):
        print("Generating data.")
        pass

    def generate_random_images(self, image, steering_angle):

        images = []
        steering_angles = []

        i = 0
        while i < GENERATED_CLONES:
            img, sa = self.__generate_random_image(image, steering_angle)

            if (sa > -THRASHOLD_STEERING_ANGLE) & (sa < THRASHOLD_STEERING_ANGLE):
                images.append(img)
                steering_angles.append(sa)
                i += 1

        print('.', end="")

        return images, steering_angles

    def random_brightness(self, image):
        return self.__random_brightness(image)

    def random_transformation(self, image, steering_angle):
        return self.__random_transformation(image, steering_angle, 100)

    def random_stripes(self, image):
        return self.__random_stripes(image, 1, 0.55, 0.1)

    def random_flip(self, image, steering_angle):
        return self.__random_flip(image, steering_angle)

    def __generate_random_image(self, image, steering_angle):

        image = self.__random_brightness(image)
        image, steering_angle = self.__random_transformation(image, float(steering_angle), GENERATOR_TRANSFORM)
        image, steering_angle = self.__random_flip(image, float(steering_angle))
        # image = self.__random_stripes(image, hls=0, mu=0.5, sigma=0.4)
        image = self.__random_stripes(image, hls=1, mu=0.55, sigma=0.1)
        # image = self.__random_stripes(image, hls=2, mu=0.55, sigma=0.4)

        return image, steering_angle

    def __random_brightness(self, image):

        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        random_brightness = 0.25 + np.random.uniform() * 0.75
        image[:, :, 2] = image[:, :, 2] * random_brightness
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        return image

    def __random_transformation(self, image, steering_angle, trans_range):

        tr_x = trans_range * np.random.uniform() - trans_range / 2
        steering_angle = steering_angle + tr_x / trans_range * 2 * .2

        tr_y = 40 * np.random.uniform() - 40 / 2
        transformation_matrix = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

        image = cv2.warpAffine(image, transformation_matrix, (320, 160))

        return image, steering_angle

    def __random_flip(self, image, steering_angle):

        if np.random.randint(2) == 1:
            image = cv2.flip(image, 1)
            steering_angle = -steering_angle

        return image, steering_angle


    def __random_stripes(self, image, hls, mu, sigma):

        width = image.shape[1]
        height = image.shape[0]

        x1 = 0
        y1 = width * np.random.uniform()
        x2 = height
        y2 = width * np.random.uniform()

        image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        shadow_mask = 0 * image[:, :, 1]

        X_mgrid = np.mgrid[0:height, 0:width][0]
        Y_mgrid = np.mgrid[0:height, 0:width][1]

        shadow_mask[((X_mgrid - x1) * (y2 - y1) - (x2 - x1) * (Y_mgrid - y1) >= 0)] = 1

        if np.random.randint(2) == 1:
            random_value = random.gauss(mu, sigma)

            masked = shadow_mask == 1
            unmasked = shadow_mask == 0

            if np.random.randint(2) == 1:
                image[:, :, hls][masked] = image[:, :, hls][masked] * random_value
            else:
                image[:, :, hls][unmasked] = image[:, :, hls][unmasked] * random_value

        image = cv2.cvtColor(image, cv2.COLOR_HLS2RGB)
        return image
