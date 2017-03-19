from parameters import *
from tools.data_preprocessor import DataPreprocessor
from tools.data_provider import DataProvider
from tools.network import Network
from keras.models import load_model
from tools.train_history import TrainingHistory

network = Network()

model = network.create_convolutional_nvidia_style_modified(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), num_classes=NUM_CLASSES)
# model = load_model('./models/T6/model-0063-0.0731.h5')

model.summary()

model.compile(loss='mse', optimizer='adam')

provider = DataProvider(DATA, LOG_FILE, TRAIN_BATCH_SIZE)
preprocessor = DataPreprocessor()
history = TrainingHistory()

X_test_raw, y_test_raw = provider.get_raw_test_data()
X_test, y_test = preprocessor.preprocess(X_test_raw, y_test_raw)

train_history = []
test_history = []

for X_train_raw, y_train_raw, count in provider.get_next_batch_of_raw_train_data():
    X_train, y_train = preprocessor.preprocess(X_train_raw, y_train_raw)

    result = model.fit(X_train, y_train,
                       batch_size=TRAIN_KERAS_BATCH_SIZE,
                       nb_epoch=TRAIN_KERAS_EPOCH,
                       validation_split=0.2,
                       shuffle=True, verbose=2)

    print('\nEvaluation on %d test samples' % X_test.shape[0])
    test_loss = model.evaluate(X_test, y_test)
    print(' - %s: %.4f' % ("loss", test_loss), end="")
    test_history.append(test_loss)

    if count % 3 == 0:
        model.save('./models/model-%.4d-%.4f.h5' % (count, test_loss))
        history.save(train_history, result, test_history, label="%.4d-%.4f" % (count, test_loss))

