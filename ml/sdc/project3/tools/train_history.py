import pickle

class TrainingHistory:
    def __init__(self):
        pass

    def save(self, history_objects, keras_history_object, test_loss_history, label):

        history_object = {}
        history_object["loss"] = keras_history_object.history['loss']
        history_object["val_loss"] = keras_history_object.history['val_loss']
        history_object["label"] = label
        history_object["test_loss"] = test_loss_history

        history_objects.append(history_object)

        pickle.dump(history_objects, open("./models/pickle-" + label + ".p", "wb" ),
                    protocol=pickle.HIGHEST_PROTOCOL)

        return history_objects

    def load_history(self, path):
        history_object = pickle.load(open(path, "rb"))

        return history_object



