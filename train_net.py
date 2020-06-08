import sys, os
import numpy as np
import pandas as pd
import neuron_net
sys.path.append('./data')
import data_loader as dl
file_path = "./" if os.getcwd()[-4:] == "data" else "./data/"

# Load datasets
# training_data, validation_data, test_data = dl.wrap_data()
training_data, validation_data, test_data = dl.load_split_dataset()
# Create and configure a network
net = neuron_net.DeepNet([784, 30, 10])
# Train the network with hyper-parameters
num_epochs = 50
batch_size = 10
learning_rate = .005
regulation_factor = .2
train_size_ratio = 25
training_data = training_data[ :len(training_data)//train_size_ratio ]
loss_types = [ "MS", "CE" ]
loss_type = loss_types[1]
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
file = "{}{}[{}_{}_{}_{}_{}].csv".format(file_path, loss_type, num_epochs, batch_size, 
                                         learning_rate, regulation_factor, train_size_ratio)
df.to_csv(file, sep=',', index=False)
print("Training complete!\n\
       See your training loss/accuracy at {}.".format(file))