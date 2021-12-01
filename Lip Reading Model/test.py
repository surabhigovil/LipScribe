import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import imutils
import time
import timeit
import itertools

from sklearn.metrics import classification_report, confusion_matrix

from train import load_data

from keras.models import model_from_json
from keras.utils import to_categorical, plot_model

np.random.seed(7)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

model_name = "model_one"
models_dir = 'lip_reading_models/'

# train_words subset of words for training
train_words = ['AGREE', 'BUSINESS', 'CUSTOMERS', 'DAVID', 'DEATH', 'ECONOMY', 'TRYING', 'UNDER', 'VICTIMS', 'WAITING', 'YEARS']
print(len(train_words))

classes = len(train_words)  # len(words)

labels = train_words

def load_model(model_path, model_weights_path):

    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(model_weights_path)
    print("Loaded Model from disk")

    # compile and evaluate loaded model
    loaded_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    print("Loaded Model Weights from disk")

    return loaded_model


model = load_model(models_dir + model_name + '.json', models_dir + model_name + '.h5')


model.summary()



X_test, y_test = load_data("NPY/", "test")


def measure_sample():
    i = np.random.choice(len(X_test), 1)[0]
    scores = model.predict(X_test[i:i+1])


total_time = 0
for x in range(0, 100):
    start_time = timeit.default_timer()
    measure_sample()
    total_time += timeit.default_timer() - start_time
print("Duration of predicting one sample: {}".format(total_time / 100))


start_time = timeit.default_timer()
scores = model.evaluate(X_test, y_test)
elapsed = timeit.default_timer() - start_time

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("Duration: {}".format(elapsed))


predictions = model.predict_classes(X_test)
lab = np.argmax(y_test, axis=1)

cm = confusion_matrix(lab, predictions)


def confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    """
    Using sklearn confusion matrix (cm)
    Source: https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    """
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('hot_r')

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(orientation="horizontal", pad=0.14, shrink=0.5)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:d}".format(int(cm[i, j] * 100)),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:d}".format(int(cm[i, j] * 100)),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual class', color="black")
    plt.xlabel('Predicted class\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), color="black")

    plt.savefig('outputs/conf_matrix_{}.pdf'.format(model_name))
    plt.show()

confusion_matrix(cm, labels, title="Confusion Matrix")
