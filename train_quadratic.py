import sys
sys.path.append('./data')
import neuron_net
import data_loader
# Load datasets
training_data, validation_data, test_data = data_loader.wrap_data()
# Create and configure a network
net = neuron_net.DeepNet([784, 30, 10])
# Train the network with hyper-parameters
num_epochs = 30
batch_size = 10
learning_rate = 3.0
regulation_factor = 0.1
# training_data = training_data[:500]
net.SGD(training_data, num_epochs, batch_size, learning_rate, regulation_factor, test_dataset=validation_data)