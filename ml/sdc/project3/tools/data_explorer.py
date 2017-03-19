import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from parameters import *

import argparse
import base64
import json

import numpy as np
import math
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

from tools.data_preprocessor import DataPreprocessor


class DataExplorer:


    def __init__(self):
        pass

    def sort_images_by_labels(self, images, labels):
        import numpy as np

        # sign_names = read_signnames()
        unique_labels = np.unique(labels)
        images_by_labels = {}

        for label in unique_labels:
            label_description = label
            indices = np.in1d(labels.ravel(), [label]).reshape(labels.shape)
            images_by_label = images[indices]
            images_by_labels[label] = (label_description, images_by_label)

        return images_by_labels


    def draw_random_image_examples_by_class(self, images_by_labels, fig_title):
        import cv2
        import numpy as np
        import random
        import matplotlib.pyplot as plt

        offset_x = 70
        offset_y = 3

        image_width = IMAGE_WIDTH
        image_height = IMAGE_HEIGHT
        border = 3
        n_rows = len(images_by_labels.keys())
        n_cols = 5

        canvas_width = offset_x + (border + image_width) * n_cols + border
        canvas_height = offset_y + (border + image_height) * n_rows + border

        canvas = np.zeros((canvas_height, canvas_width, 3), np.uint8)
        canvas[::] = 1.
        canvas = np.array(canvas).astype('float32')

        fig = plt.figure(figsize=(30,80))
        ax = plt.gca()
        ax.set_title(fig_title)
        ax.axis("off")

        j = 0

        sorted_keys = []
        for key in images_by_labels.keys():
            sorted_keys.append(int(key))

        sorted_keys.sort()

        for key in sorted_keys:
            desc = images_by_labels[key][0]
            imgs = images_by_labels[key][1]

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(canvas, str(desc), (20, ((border + image_height) * j) + image_height - 4), font, 0.5, (0, 0, 0), 1)

            for i in range(n_cols):
                index = random.randint(0, imgs.shape[0] - 1)
                s_img = imgs[index]

                canvas[offset_y + ((border + image_height) * j):offset_y + ((border + image_height) * j) + image_height,
                offset_x + ((border + image_width) * i):offset_x + ((border + image_width) * i) + image_width] = s_img

            j += 1

        plt.imshow(canvas,cmap="gray")
        plt.show()


    def show_softmax_probabilities_by_class(self, images_by_labels, fig_title, file_model):

        model = load_model(file_model)

        for label in range(-24, 25):

            if label in images_by_labels:

                fig = plt.figure(figsize=(18, 2))

                img = images_by_labels[label][1][0]

                pp = DataPreprocessor()
                x = pp.preprocess_to_drive([img])

                y = model.predict(x, batch_size=1)

                plt.bar(list(range(-24, 25)), y.tolist()[0])

                plt.title(label)

                height = 10
                plt.xticks(np.arange(-24, 24, 1.0), list(range(-24,25)))
                # plt.yticks(np.arange(0.0, 1.0, 0.1), np.arange(0.0, 1.0, 0.1))

                ax = plt.axes([.8, 0.25, 0.5, 0.5], frameon = True)
                ax.imshow(img)
                ax.axis('off')

        plt.show()



    def draw_data_distribution_by_class(self, images_by_labels, fig_title):

        N = len(images_by_labels.keys())

        menMeans = (20, 35, 30, 35, 27)

        sorted_keys = []
        for key in images_by_labels.keys():
            sorted_keys.append(int(key))

        sorted_keys.sort()

        numOfImages = []
        xLabels = []
        for key in sorted_keys:
            numOfImages.append(len(images_by_labels[key][1]))
            xLabels.append(images_by_labels[key][0])

        menMeans = tuple(numOfImages)

        ind = np.arange(N)  # the x locations for the groups
        width = .8       # the width of the bars

        fig, ax = plt.subplots(figsize=(16,10))

        rects1 = ax.bar(ind, menMeans, width, color='r')

        # add some text for labels, title and axes ticks
        ax.set_ylabel('Number of training examples')
        ax.set_title(fig_title)
        ax.set_xticks(ind + width/2)

        ax.set_xticklabels(tuple(xLabels))
        ax.yaxis.grid(True)

        # ax.legend((rects1[0]), ('Men'))
        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=90, fontsize=10)

        labels = ax.get_yticklabels()
        plt.setp(labels, fontsize=10)

        def autolabel(rects):
            # attach some text labels
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                        '%d' % int(height),
                        ha='center', va='bottom', fontsize=7)

        autolabel(rects1)

        plt.show()


    def plot_image(self, image, fig_title=""):
        import random
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 2.5))
        ax = plt.gca()
        ax.set_title(fig_title)

        # if cv2_image == True:
        #     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap="gray")
        # else:
        plt.imshow(image, cmap="gray")

        plt.show()


    def plot_3(self, list_of_images_with_titles):
        rows = 1
        cols = 3

        f, axes = plt.subplots(rows, cols, figsize=(14, 9))
        f.tight_layout()

        axes[0].imshow(list_of_images_with_titles[0][0])
        axes[0].set_title(list_of_images_with_titles[0][1])

        axes[1].imshow(list_of_images_with_titles[1][0])
        axes[1].set_title(list_of_images_with_titles[1][1])

        axes[2].imshow(list_of_images_with_titles[2][0])
        axes[2].set_title(list_of_images_with_titles[2][1])


    def plot_loss_of_training_dataset(self, training_history):

        total = len(training_history)

        rows = math.ceil(math.sqrt(total))
        cols = rows

        fig = plt.figure(figsize=(14,9))

        gs1 = gridspec.GridSpec(rows, cols)
        gs1.update(wspace=.5, hspace=1.)  # set the spacing between axes.

        for i in range(total):
            if i >= total:
                return

            axes = plt.subplot(gs1[i])

            figure = training_history[i]["label"]
            loss = training_history[i]["loss"]
            val_loss = training_history[i]["val_loss"]

            axes.set_title('Batch-TestLoss = ' + figure, fontsize=7)
            axes.plot(loss)
            axes.plot(val_loss)
            axes.legend(['train', 'val'], prop={'size': 6})
            axes.set_xlabel("EPOCH", fontsize=7)
            axes.set_ylabel("MSE", fontsize=7)

        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)


    def plot_loss_of_test_dataset(self, training_history):

        fig = plt.figure(figsize=(10, 6))

        plt.plot(training_history[0]["test_loss"])
        plt.title('MSE of Test Data Set')
        plt.legend(['test data set'])
        plt.ylabel('MSE')
        plt.xlabel('batch')
        plt.grid()
        plt.show()
        pass



