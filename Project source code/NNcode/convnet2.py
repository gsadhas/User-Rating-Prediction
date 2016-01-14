import pickle

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import Imputer

import models2


def load2d(x_train, image_dim):
    # return reshaped array
    return x_train.reshape(x_train.shape[0], 1, x_train.shape[1]).astype("float32")


def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy
    '''
    y = np.asarray(y, dtype='int32')

    # if not nb_classes:
    #     nb_classes = np.max(y) + 1

    Y = np.zeros((len(y), 5))
    for i in range(0, len(y)):
        Y[i, y[i]-1] = 1.
    return Y

print "Loading train data.."
# load data
x = pickle.load(open("data/trainFeatures.p", "rb"))
y = pickle.load(open("data/trainLabels.p", "rb"))
y = to_categorical(y)
print y.shape

dim = 1
# x = load2d(np.asarray(x), x.shape[1],2)
x = load2d(np.asarray(x.toarray()), dim)
print "Train shape", x.shape


# for i, x1 in enumerate(x):
#     if x1.shape[0] > 1:
#         print i, x1

x_test = pickle.load(open("data/testFeatures.p", "rb"))

x_test = load2d(np.asarray(x_test.toarray()), dim)
print "Test shape", x_test.shape

y_test = pickle.load(open("data/testLabels.p", "rb"))
print y_test.shape

# get all test images along with their names
print "Loading test data.."
# get model
print "compiling model.."
model = models2.get_cnn_kmodel(x.shape[2], x.shape[1])
print "fitting data"
model.fit(x, y, nb_epoch=15, validation_split=0.05, show_accuracy=True,
          verbose=1, batch_size=32, shuffle=True)
print "saving model.."
models2.save_cnn_model(model, "data/models/cnn5k")
# testing
# model = models2.load_cnn_model(model_arch="data/models/cnn5kcnn_arch.json",model_weights="data/models/cnn5kcnn_weights")

print "Testing"
preds = model.predict_classes(x_test, verbose=2)

test_preds = []
for p in preds:
    test_preds.append(p+1)
print test_preds

preds = test_preds

print metrics.confusion_matrix(y_test, preds)
