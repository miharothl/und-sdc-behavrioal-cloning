import cv2
import numpy as np
import random

from parameters import *
from sklearn.utils import shuffle
from tools.data_generator import DataGenerator
from tools.data_preprocessor import DataPreprocessor


class DataProvider:

    def __init__(self, paths_to_data, log_file, train_batch_size):
        self.__preprocessor = DataPreprocessor()

        log_raw = self.__load_log_raw(paths_to_data, log_file)
        log = self.__preprocess_raw_log(log_raw)

        log_train, log_test = self.__shuffle_and_split_log(log)

        self.__log_train_batches = self.__create_log_batches(train_batch_size, log_train)
        self.__log_test = log_test

        pass

    def get_next_batch_of_raw_train_data(self):
        i = 0
        X, y = [], []

        while i < len(self.__log_train_batches):

            print("\n\nGetting train batch %d/%d." % (i, len(self.__log_train_batches)), end=" ")

            batch = self.__log_train_batches[i][0]
            X, y = self.__extraxt_raw_features_and_labels(batch)

            yield X, y, i
            i += 1

    def get_random_log_raw_pictures(self, paths_to_data, log_file, train_batch_size):
        log_raw = self.__load_log_raw(paths_to_data, log_file)

        log_raw_record = log_raw[random.randint(0, len(log_raw)-1)]
        log_raw = [log_raw_record]

        log = self.__preprocess_raw_log(log_raw)

        X = []
        y = []

        for record in log:
            filename = self.__remove_windows_path(record[0])

            image = cv2.imread(filename, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            steering_angle = record[1]

            X.append(image)
            y.append(steering_angle)

        return X, y

    def get_raw_test_data(self):
        print("\n\nGetting test data...")
        return self.__extraxt_raw_features_and_labels(self.__log_test, enable_generation=False)

    def __extraxt_raw_features_and_labels(self, log, enable_generation=True):
        X = []
        y = []

        if enable_generation:
            data_generator = DataGenerator()

        for record in log:

            filename = self.__remove_windows_path(record[0])

            image = cv2.imread(filename, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            steering_angle = record[1]

            if SUPPRESS_LABEL_0:
                label = self.__preprocessor.steering_angle_to_label(np.array([float(steering_angle)]))
                if label[0] == 0:
                    continue

            X.append(image)
            y.append(steering_angle)

            if enable_generation:
                generated_images, generated_steering_angles = data_generator.generate_random_images(image, steering_angle)

                for i in range(0, len(generated_images)):
                    X.append(generated_images[i])
                    y.append(generated_steering_angles[i])

        if enable_generation:
            print("\n")

        return X, y

    def __preprocess_raw_log(self, raw_log):

        log = []

        for record in raw_log:

            file_center = record[0]
            file_left = record[1]
            file_right = record[2]
            steering_angle = record[3]

            random_gauss_adjustment = random.gauss(STEERING_ANGLE_ADJUSTMENT_MU, STEERING_ANGLE_ADJUSTMENT_SIGMA)

            if file_center != '':
                log.append([file_center, float(steering_angle)])

            if file_left != '':
                adjusted_steering_angle = float(steering_angle) + random_gauss_adjustment
                if adjusted_steering_angle > THRASHOLD_STEERING_ANGLE:
                    adjusted_steering_angle = THRASHOLD_STEERING_ANGLE
                log.append([file_left, adjusted_steering_angle])

            if file_right != '':
                adjusted_steering_angle = float(steering_angle) - random_gauss_adjustment
                if adjusted_steering_angle < -THRASHOLD_STEERING_ANGLE:
                    adjusted_steering_angle = -THRASHOLD_STEERING_ANGLE
                log.append([file_right, adjusted_steering_angle])

        log = np.array(log)

        print('\nLog:')
        print(' - total images : %d' % log.shape[0])

        return log

    def __decision(self, probability):
        return random.random() > probability

    def __load_log_raw(self, paths_to_data, log_file):

        count_center = count_left = count_right = count_discarded = 0
        log = []
        for path_to_data in paths_to_data:
            path = path_to_data[0]
            label_filter = path_to_data[1]

            import csv
            with open(path + log_file, 'r') as csvfile:
                next(csvfile)  # skip header
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:

                    label = self.__preprocessor.steering_angle_to_label(np.array(row[3]).astype(float))

                    if label in label_filter:
                        probability = label_filter[label]
                        if self.__decision(probability):
                            count_discarded += 1
                            continue

                    row[0] = path + row[0]
                    count_center += 1

                    if row[1] != '':
                        row[1] = path + row[1].lstrip()
                        count_left += 1

                    if row[2] != '':
                        row[2] = path + row[2].lstrip()
                        count_right += 1

                    if (float(row[6]) > THRESHOLD_SPEED)\
                            & (float(row[3]) <= THRASHOLD_STEERING_ANGLE)\
                            & (float(row[3]) >= -THRASHOLD_STEERING_ANGLE):
                        log.append(row)
                    else:
                        count_discarded += 1

        print('\nRAW log:')
        print(' - center: %d' % count_center)
        print(' - left: %d' % count_left)
        print(' - right: %d' % count_right)
        print(' - discarded: %d' % count_discarded)

        return np.array(log)

    def __shuffle_and_split_log(self, log):

        log = shuffle(log)

        len(log)
        idx_train = int(len(log) * (1.-TEST_DATA))

        log_train = log[:idx_train]
        log_test = log[idx_train:]

        print('\nData:')
        print(' - train: %d' % log_train.shape[0])
        print(' - test: %d' % log_test.shape[0])

        return log_train, log_test

    def __create_log_batches(self, batch_size, log):
        batches = []

        sample_size = len(log)
        for start_i in range(0, sample_size, batch_size):
            end_i = start_i + batch_size
            batch = [log[start_i:end_i]]
            batches.append(batch)

        return batches

    def __remove_windows_path(self, filename):

        index_c = filename.find('C:\\')
        index_d = filename.find('D:\\')

        if index_c == -1 & index_d == -1:
            return filename

        win_begin = -1
        if index_c >= 0:
            win_begin = index_c
        else:
            win_begin = index_d

        win_end = filename.find('IMG')

        new_filename = filename[:win_begin] + 'IMG/' + filename[win_end+4:]

        return new_filename


