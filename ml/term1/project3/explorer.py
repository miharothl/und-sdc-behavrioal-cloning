from parameters import *
from tools.data_generator import DataGenerator
from tools.data_provider import DataProvider
from tools.data_preprocessor import DataPreprocessor
from tools.data_explorer import DataExplorer
from tools.train_history import TrainingHistory

history = TrainingHistory()
explorer = DataExplorer()

history_objects = history.load_history('model.p')
explorer.plot_loss_of_training_dataset(history_objects)
explorer.plot_loss_of_test_dataset(history_objects)







provider = DataProvider(DATA, LOG_FILE, TRAIN_BATCH_SIZE)

X, y = provider.get_random_log_raw_pictures(DATA, LOG_FILE, TRAIN_BATCH_SIZE)

explorer = DataExplorer()

explorer.plot_image(X[0], "Center Camera - Steering Angle = %0.2f" % float(y[0]))
explorer.plot_image(X[1], "Left Camera - Steering Angle Adjusted = %0.2f" % float(y[1]))
explorer.plot_image(X[2], "Right Camera - Steering Angle Adjusted = %0.2f" % float(y[2]))

generator = DataGenerator()
image = X[0]
steering_angle = float(y[0])

image_generated = generator.random_brightness(image)
explorer.plot_image(image, "Original")
explorer.plot_image(image_generated, "Random Brightness")

gen_img, gen_sa = generator.random_flip(image, steering_angle)
explorer.plot_image(gen_img, "Flipped - Steering Angle = %0.2f" % gen_sa)
gen_img, gen_sa = generator.random_flip(image, steering_angle)
explorer.plot_image(gen_img, "Flipped - Steering Angle = %0.2f" % gen_sa)
gen_img, gen_sa = generator.random_flip(image, steering_angle)
explorer.plot_image(gen_img, "Flipped - Steering Angle = %0.2f" % gen_sa)

gen_img, gen_sa = generator.random_transformation(image, steering_angle)
explorer.plot_image(gen_img, "Transformed - Steering Angle Adjusted = %0.2f" % gen_sa)
gen_img, gen_sa = generator.random_transformation(image, steering_angle)
explorer.plot_image(gen_img, "Transformed - Steering Angle Adjusted = %0.2f" % gen_sa)
gen_img, gen_sa = generator.random_transformation(image, steering_angle)
explorer.plot_image(gen_img, "Transformed - Steering Angle Adjusted = %0.2f" % gen_sa)

gen_img = generator.random_stripes(image)
explorer.plot_image(gen_img, "Random Stripe")
gen_img = generator.random_stripes(image)
explorer.plot_image(gen_img, "Random Stripe")
gen_img = generator.random_stripes(image)
explorer.plot_image(gen_img, "Random Stripe")







for X_train_raw, y_train_raw, count in provider.get_next_batch_of_raw_train_data():
    X = X_train_raw
    y = y_train_raw
    break

preprocessor = DataPreprocessor()
X_train, y_train = preprocessor.preprocess(X_train_raw, y_train_raw)

explorer = DataExplorer()
images_by_labels = explorer.sort_images_by_labels(X_train, y_train)

explorer.show_softmax_probabilities_by_class(images_by_labels, "softmax")


# explorer.draw_data_distribution_by_class(images_by_labels, "train data 41")
#
# explorer.draw_random_image_examples_by_class(images_by_labels, "train data 41")

the = 'end'

