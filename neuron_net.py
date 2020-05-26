import numpy as np

class DeepNet(object):
    # yi = Wi * xi + bi
    def __init__(self, sizes):
        # Number of layers in the network
        self.num_layers = len(sizes)
        # Number of neuron in the network
        self.num_neuron = sizes
        # Populate Gaussian random of bias vectors, layer by layer
        self.bs = [ np.random.rand(r, 1) for r in sizes[1:] ]
        # Populate Gaussian random of weight matrix, layer by layer
        self.Ws = [ np.random.rand(r, c) for r, c in zip(sizes[1:], sizes[:-1]) ]
    
    # Function computes the sigmoid neutron
        # Input: weighted input vector of a layer
        # Output: normalized value of weighted input vector of the same layer
    def sigma(z):
        return 1.0 / (1 + np.exp(-z))
    
    # Function computes the derivative of sigmoid neutron
        # Input: sigmoid neutron z (normalized )
        # Output: rate of change in sigmoid neutron z
    def sigma_rate(self, z):
        return sigma(z) * (1 - sigma(z))
    
    # Function computes the fordward activation of a layer:
        # Input: activation vector a of a current layer
        # Output: activation vector for the next layer (forward activation)    
    def feedforward(self, a):
        for b, W in zip(self.bs, self.Ws):
            # Compute weighted input
            a = sigma(np.dot(W, a) + b)
        return a
   
    # Function backpropagation
        # Input: a data sample x, y
        # Output: a tuple of (gd_bs, gd_Ws) representing the gradient for the loss function
        # gd_bs, with same dimension to bs', is list of bias vectors, layer by layer
        # gd_Ws, with same dimension to Ws', is list of weight matrices, layer by layer
    def backpropagation(self, x, y):
        # Populate vectors in gd_bs with 0 layer by layer
        gd_bs = [ np.zeros(b.shape) for b in self.bs ]
        # Populate matrices in gd_Ws with 0 layer by layer
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
            a = sigma(z)
            activations.append(a)
        
        # Back propagation
        delta = (activations[-1] - y) * sigma_rate(zs[-1])
        gd_bs[-1], gd_Ws[-1]= delta, np.dot(delta, activations[-2].transposes())
        for k in range(2, self.num_layers):
            z, s = zs[-k], sigma_rate(z)
            delta = np.dot(self.Ws[-k + 1].transpose(), delta) * s
            gd_bs[-k], gd_Ws[-k] = delta, np.dot(delta, activations[-k - 1].transposes())
        
        return (gd_bs, gd_Ws)
        
    # Function evaluate:
        # Input: test data in tuple of (x, y) 
        # Output: number of correct predictions
    def evaluate(self, test):
        results = [ (np.argmax(self.forwardfeed(x)), y) for (x, y) in test ]
        return sum(int(y0 == y1) for (y0, y1) in results)
    
    # Function update bias vectors and weight matrices, layer by layer 
        # Input: a batch of mini samples mini_batch, and learning rate eta
        # Output: None, just update the network's bias vectors bs and the weight matrix Ws,
        # layer by layer using gradient descent and backpropagation algorithm 
        # applied to the mini batch with following formulars:
        # new W = current W - eta * change in loss function per change in weight
        # new b = current b - eta * change in loss function per change in weight
    def update_params(self, mini_batch, eta):
        # Populate vectors in gd_bs with 0 layer by layer
        gd_bs = [ np.zeros(b.shape) for b in self.bs ]
        # Populate matrices in gd_Ws with 0 layer by layer
        gd_WS = [ np.zeros(W.shape) for W in self.Ws ]
        for x, y in mini_batch:
            # Compute delta bias bs and delta weights Ws
            dt_bs, dt_Ws = backpropagation(x, y)
            # Update vectors of gradient in bias for the loss function
            gd_bs = [ gd_b + dt_b for gd_b, dt_b in zip(gd_bs, dt_bs) ]
            # Update matrices of gradient in weight for the loss function
            gd_Ws = [ gd_W + dt_W for gd_W, dt_W in zip(gd_Ws, dt_Ws) ]
        
        # Update bias vectors in the network, layer by layer
        self.bs = [ b - (eta / len(mini_batch)) * gd_b for b, gd_b in zip(self.bs, gd_bs) ]
        # Update weight matrices in the network, layer by layer
        self.Ws = [ W - (eta / len(mini_batch)) * gd_W for W, gd_W in zip(self.Ws, gd_Ws) ]
    
    # Function Stochastic Gradient Descent (SGD)
    # Input: training dataset, number of epochs, size of mini batch, learning rate, 
    # optional testing dataset
    # Output: None, just print out the training progress per every epoch
    def SGD(self, train_dataset, epochs, batch_size, eta, test_dataset=None):
        for k in range(epochs):
            np.random.shuffle(train_dataset)     # randomize samples in training data
            batch = [ train_dataset[ k : batch_size ]
                      for k in range(0, len(train_dataset), batch_size) ]
            for sample in batch:
                self.update_params(sample, eta)
            if test_dataset:
                print ("Epoch {0}: {1} / {2} ...".format(k, 
                                                         self.evaluate(test_dataset), 
                                                         len(test_dataset)))
            else:
                print ("Epoch {0} complete ...".format(k))
