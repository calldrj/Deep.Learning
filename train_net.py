import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.append('./data')

file_path = "./"
if os.getcwd()[-4:] != "data":
    file_path = "./data/"

import neuron_net
import data_loader
# Load datasets
training_data, validation_data, test_data = data_loader.wrap_data()
# Create and configure a network
net = neuron_net.DeepNet([784, 30, 10])
# Train the network with hyper-parameters
num_epochs = 400
batch_size = 10
learning_rate = 0.5
regulation_factor = 0.1
training_data = training_data[:1000]
loss_types = [ "MSE", "CE" ]
loss_type = loss_types[0]
loss_train, loss_validation, \
accuracy_train, accuracy_validation = net.SGD(training_data, num_epochs, batch_size, 
                                              learning_rate, regulation_factor, 
                                              loss_type, test_dataset=validation_data)
# Create a data frame to save the loss and accuracy data
df = pd.DataFrame({ "epoch":                np.arange(1, num_epochs + 1, 1),
                    "loss_train":           loss_train, 
                    "loss_validation":      loss_validation, 
                    "accuracy_train":       accuracy_train , 
                    "accuracy_validation":  accuracy_validation })
# Write the data frame in local drive
df.to_csv("{}neuron_net_{}_{}.csv".format(file_path, num_epochs, loss_type), sep=',', index=False)
