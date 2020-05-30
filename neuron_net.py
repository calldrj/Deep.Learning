import numpy as np
import random

class DeepNet(object):
    # Object is a list of input size, number of neurons in the kth layer, ..., and output size
    # For example, net = neuron_net.DeepNet([784, 30, 10])
    # y = W * x + b
    def __init__(self, sizes):
        # Number of layers in the network
        self.num_layers = len(sizes)
        # Number of neuron in the network
        self.num_neuron = sizes
        # Populate Gaussian random on a list of bias vectors, each layer is represented by a vector b
        self.bs = [ np.random.randn(r, 1) for r in sizes[1:] ]
        # Populate Gaussian random on a list of weight matrices, each matrix is repesented by a matrix M
        self.Ws = [ np.random.randn(r, c) for r, c in zip(sizes[1:], sizes[:-1]) ]
    
    # Function computes the output of the network, given an input:
        # Input:  an input array a of the first layer (size arbitrarily nx1)
        # Output: an out array of the network (size arbitrarily mx1)   
    def feedforward(self, a):
        for b, W in zip(self.bs, self.Ws):
            # Compute weighted input for every layer, Numpy applied sigmoid() elementwise
            a = sigmoid(np.dot(W, a) + b)
        return a

     # Function evaluate:
        # Input: test data in tuple of (x, y) 
        # Output: number of correct predictions
    def evaluate(self, test):
        c = 0
        for (x, y) in test:
            c += (int)(np.argmax(self.feedforward(x)) == y) 
        return c
   
    # Function backpropagation
        # Input: a data sample x, y
        # Output: a tuple of (gd_bs, gd_Ws) representing the gradient for the loss function
        # gd_bs, with same dimension to bs', is list of bias vectors, layer by layer
        # gd_Ws, with same dimension to Ws', is list of weight matrices, layer by layer
    def backpropagation(self, x, y):
        # Populate list of vectors gd_bs each with vector 0, layer by layer
        gd_bs = [ np.zeros(b.shape) for b in self.bs ]
        # Populate list of matrices gd_Ws each with matrix 0, layer by layer
        gd_Ws = [ np.zeros(W.shape) for W in self.Ws ]
        
        # Feedforward
        a = x               # input vector (the 1st layer)
        activations = [x]   # list of all activation vectors from the 1st to the last layers
        zs = []             # list of all weighted input vectors from the 2nd to the last layers
        
        for b, W in zip(self.bs, self.Ws):
            # Compute the individual weighted input vector, layer by layer, then save in zs
            z = np.dot(W, a) + b
            zs.append(z)
            # Compute the forward activation a, layer by layer, then save in activations
            a = sigmoid(z)
            activations.append(a)
        
        # Back propagation
        delta = (activations[-1] - y) * sigmoid_rate(zs[-1])
        gd_bs[-1], gd_Ws[-1] = delta, np.dot(delta, activations[-2].transpose())
        for k in range(2, self.num_layers):
            z = zs[-k]
            s = sigmoid_rate(z)
            delta = np.dot(self.Ws[-k + 1].transpose(), delta) * s
            gd_bs[-k], gd_Ws[-k] = delta, np.dot(delta, activations[-k - 1].transpose())
        
        return (gd_bs, gd_Ws)
         
    # Function update bias vectors and weight matrices, layer by layer 
        # Input: a batch of samples mini_batch, and learning rate eta
        # Output: None, just update the network's bias vectors bs and the weight matrix Ws,
        # layer by layer using gradient descent and backpropagation algorithm 
        # applied to the mini batch with following formulars:
        # new W = current W - eta * change in loss function per change in weight
        # new b = current b - eta * change in loss function per change in weight
    def update_params(self, mini_batch, eta):
        # Populate list of vectors gd_bs each with vector 0, layer by layer
        gd_bs = [ np.zeros(b.shape) for b in self.bs ]
        # Populate list of matrices gd_Ws each with matrix 0, layer by layer
        gd_Ws = [ np.zeros(W.shape) for W in self.Ws ]
        for x, y in mini_batch:
            # Compute delta bias bs and delta weights Ws
            dt_bs, dt_Ws = self.backpropagation(x, y)
            # Update vectors of gradient in bias for the loss function
            gd_bs = [ gd_b + dt_b for gd_b, dt_b in zip(gd_bs, dt_bs) ]
            # Update matrices of gradient in weight for the loss function
            gd_Ws = [ gd_W + dt_W for gd_W, dt_W in zip(gd_Ws, dt_Ws) ]
        
        # Update list of bias vectors in the network, layer by layer
        self.bs = [ b - (eta / len(mini_batch)) * gd_b for b, gd_b in zip(self.bs, gd_bs) ]
        # Update list of weight matrices in the network, layer by layer
        self.Ws = [ W - (eta / len(mini_batch)) * gd_W for W, gd_W in zip(self.Ws, gd_Ws) ]
    
    # Function Stochastic Gradient Descent (SGD)
    # Input: training dataset, number of epochs, size of mini batch, learning rate, 
    # optional testing dataset
    # Output: None, just print out the training progress per every epoch
    def SGD(self, train_dataset, epochs, batch_size, eta, test_dataset=None):
        # Unzip train_dataset 
        train_dataset = list(train_dataset)
        l = len(train_dataset)
        # Process test dataset if it is input
        if test_dataset:
            test_dataset = list(test_dataset)
            n = len(test_dataset)
        # Run SGD algorithms in each epoch
        for k in range(epochs):
            # Shuffle samples in training data, then partition it in given batch size
            random.shuffle(train_dataset)     
            batches = [ train_dataset[ j : j + batch_size ] for j in range(0, l, batch_size) ]
            # Run SGD algorithm in each sample of the partition
            for mini_batch in batches:
                self.update_params(mini_batch, eta)
            if test_dataset:
                print("Epoch #{}:\t{} / {} ...".format(k + 1, self.evaluate(test_dataset), n))
            else:
                print("Epoch #{}\tcomplete ...".format(k + 1))
        
        print("Training complete!")

# Function computes the sigmoid neutron
    # Input: weighted input vector of a layer z
    # Output: normalized value of weighted input vector of the same layer
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
    
# Function computes the derivative of sigmoid neutron
    # Input: sigmoid neutron z (normalized )
    # Output: rate of change in sigmoid neutron z
def sigmoid_rate(z):
    return sigmoid(z) * (1.0 - sigmoid(z))