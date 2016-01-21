
from pylearn2.utils import serial
from theano import tensor as T
from theano import function
import scipy.io
from sklearn import preprocessing
import numpy as np


def percentage(part, whole):
    return 100 * float(part)/float(whole)

model_path = 'timit_model.pkl'
model = serial.load(model_path)

X = model.get_input_space().make_theano_batch()
Y = model.fprop(X)

Y = T.argmax(Y, axis=1)

f = function([X], Y, allow_input_downcast=True)

# Load data (test set)
mat = scipy.io.loadmat('../../Datasets/Vowels_PLP_273/test_set_273.mat')
test_set = mat['test_set']
test_labels = scipy.io.loadmat('../../Datasets/Vowels_PLP_273/test_labels_273.mat')
test_labels = test_labels['test_labels']

# Convert labels to integers
le = preprocessing.LabelEncoder()
test_labels = np.ravel(test_labels)
le.fit(test_labels)
test_labels = le.transform(test_labels)

y = f(test_set)

# Difference between labels and predicted values
result = np.subtract(test_labels, y)
count_non_zero = np.count_nonzero(result)
count_zero = result.shape[0] - count_non_zero
accuracy = percentage(count_zero, result.shape[0])
print(accuracy)

a = 0