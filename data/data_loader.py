import pickle, gzip, urllib.request
import numpy as np
import random
import os

file_path = "./"
if os.getcwd()[-4:] != "data":
    file_path = "./data/"

# Function load raw dataset
def load_dataset():
    # Read data drom  source, just read once!!!
    # urllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "./data/mnist.pkl.gz")
    file = gzip.open("{}mnist.pkl.gz".format(file_path), 'rb')
    train_data, val_data, test_data = pickle.load(file, encoding='latin1')
    file.close()
    return (train_data, val_data, test_data)

# Function vectorize ouput y in training dataset
def vector_y(k):
    y = np.zeros((10, 1))
    y[k] = 1.0
    return y 

# Function process and split raw dataset
def wrap_data():
    # Load data in 3 collections
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
    return (list(train_dataset), list(val_dataset), list(test_dataset))

# Transform the raw datasets to csv format
def trans_csv():
    # Load data in 3 collections
    train_set, val_set, test_set = load_dataset()
    partitions = [ ("train", train_set ), ("validation", val_set), ("test", test_set) ]
    for name, partition in partitions:
        print("{} dataset: {} {} ...".format(name, partition[0].shape, partition[1].shape))
        features = [ f.tolist() for f in partition[0] ]
        labels   = [ l.tolist() for l in partition[1] ]
        if name == "test":
            samples = features
        else:
            samples = np.insert(features, 0, labels, axis=1)
        # Save datasets in csv format
        np.savetxt("{}{}.csv".format(file_path, name), samples, delimiter=',')
    
    print("Process complete!")

# Generate synthetic dataset
# Input:  vector of weights with size 1xm for features
#         scalar value for  bias,
#         number of samples of the dataset
#         scalar value for noise variance
# Output: an matrix of feature X, and a vector of target y such that y = Xw + b + noise   
def synthetic_data(weights, bias, num_samples, noise_variance):
    # Generate a matrix (n, m) of values with normal distribution (0, 1)
    X = np.random.normal(0, 1, (num_samples, len(weights)))
    # Compute vector y = Xw + b
    y = np.dot(X, weights) +  bias
    # Add noise with given variance
    y += np.random.normal(0, noise_variance, y.shape)
    return X, y