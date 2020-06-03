import sys
import os
import json
import numpy as np
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
regulation_factor = 0.01
training_data = training_data[:1000]
loss_type = [ "MSE", "CE" ]

loss_train, loss_validation, \
accuracy_train, accuracy_validation = net.SGD(training_data, num_epochs, batch_size, 
                                              learning_rate, regulation_factor, 
                                              loss_type[1], test_dataset=validation_data)

partitions = [ ("loss_train", loss_train), ("loss_validation", loss_validation ), 
               ("accuracy_train", accuracy_train ), ("accuracy_validation", accuracy_validation) ]
for filename, partition in partitions:
    # Save datasets in csv format
    print("Saving {} in {}".format(filename, file_path))
    np.savetxt("{}{}.csv".format(file_path, filename), partition, delimiter=',')