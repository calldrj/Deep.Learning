import pickle
import gzip
import numpy as np

# Function load raw dataset
def load_dataset():
    file = gzip.open('../data/mnist.pkl.gz', 'rb')
    train_data, val_data, test_data = pickle.load(file, encoding='latin1')
    file.close()
    return (train_data, val_data, test_data)

# Function vectorize ouput y
def vector_y(k):
    y = np.zeros((10, 1))
    y[k] = 1.0
    return y 

# Function process and split raw dataset
def wrapper_data():
    train_set, val_set, test_set = load_dataset()
    # Create training dataset
    train_x = [ np.reshape(x, (784, 1)) for x in train_set[0] ]
    train_y = [ vector_y(y) for y in train_set[1] ]
    train_dataset = zip(train_x, train_y)
    # Create validation dataset
    val_x = [ np.reshape(x, (784, 1)) for x in val_set[0] ]
    val_dataset = zip(val_x, val_set[1])
    # Create testing dataset
    test_x = [ np.reshape(x, (784, 1)) for x in test_set[0] ]
    test_dataset = zip(test_x, test_set[1])
    return (train_dataset, val_dataset, test_dataset)