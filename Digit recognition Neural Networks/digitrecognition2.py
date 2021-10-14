# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 14:35:21 2021

@author: SREENEHA
"""

#### Libraries
# Standard library
import random
import _pickle as cPickle
import gzip



# Third-party libraries
import numpy as np
    

def CrossEntropyfn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

   
def CrossEntropydelta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.
        """
        return (a-y)



def load_data():
    
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f, encoding='iso-8859-1')
    f.close()
    return (training_data, validation_data, test_data)


def default_weight_initializer(sizes):
        
        biases = [np.random.randn(y, 1) for y in sizes[1:]]
        weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        return biases, weights
    

def load_data_wrapper():
    
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e



def __init__(sizes):
      
        num_layers = len(sizes)
        sizes = sizes
        biases, weights = default_weight_initializer(sizes) 
        #biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #weights = [np.random.randn(y, x)
                        #for x, y in zip(sizes[:-1], sizes[1:])]
        return(biases, weights, num_layers)

def feedforward(a, biases, weights):
        
        for b, w in zip(biases, weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    
def accuracy(data, biases, weights, convert=False):
       
        if convert:
            results = [(np.argmax(feedforward(x, biases, weights)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(feedforward(x, biases, weights)), y) for (x, y) in data]
        
        val= sum(int(x == y) for (x, y) in results)
        #print(val)
        return val
    


def SGD(training_data, epochs, mini_batch_size, eta, num_layers, biases, weights, lmbda = 0.0,
            evaluation_data=None,
            test_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            monitor_testing_accuracy=False):
       
        
        
        if evaluation_data: n_data = len(evaluation_data)
        if evaluation_data: ntest_data = len(test_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                weights, biases=update_mini_batch(mini_batch, eta, biases, weights, num_layers, lmbda, len(training_data))
            #print("j=:", j)
            print("Epoch {0} complete".format(j))
            if monitor_training_cost:
                cost = total_cost(training_data, lmbda, biases, weights)
                training_cost.append(cost)
                print ("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                acc=accuracy(training_data, biases, weights, convert=True)
                training_accuracy.append(acc)
                print ("Accuracy on training data: {} / {}".format(acc, n))
            if monitor_evaluation_cost:
                cost = total_cost(evaluation_data, lmbda, biases, weights, convert=True)
                evaluation_cost.append(cost)
                print ("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                acc = accuracy(evaluation_data, biases, weights)
                evaluation_accuracy.append(acc)
                print ("Accuracy on evaluation data: {} / {}".format(accuracy(evaluation_data, biases, weights), n_data))
            if monitor_testing_accuracy:
                acc = accuracy(test_data, biases, weights)
                evaluation_accuracy.append(acc)
                print ("Accuracy on test data: {} / {}".format(accuracy(test_data, biases, weights), ntest_data))
       

def update_mini_batch(mini_batch, eta, biases, weights, num_layers, lmbda, n):
        
        nabla_b = [np.zeros(b.shape) for b in biases]
        nabla_w = [np.zeros(w.shape) for w in weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = backprop(x, y, biases, weights, num_layers)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(weights, nabla_w)]
        biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(biases, nabla_b)]
        return(weights, biases)

def backprop(x, y, biases, weights, num_layers):
        
        nabla_b = [np.zeros(b.shape) for b in biases]
        nabla_w = [np.zeros(w.shape) for w in weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(biases, weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = CrossEntropydelta(zs[-1], activations[-1], y)    
        delta = cost_derivative(activations[-1], y) *sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    

    
def total_cost(data, lmbda, biases, weights, convert=False):
        
        cost = 0.0
        for x, y in data:
            a = feedforward(x, biases, weights)
            if convert: y = vectorized_result(y)
            cost += CrossEntropyfn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in weights)
        return cost
    

def cost_derivative(output_activations, y):
    return(output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))



training_data, validation_data, test_data = load_data_wrapper()

#print("after load wrapper")
#print(len(training_data))

biases, weights, num_layers= __init__([784, 100, 10])

#print ("No of Layers", num_layers)

#net = Network([784, 30, 10])

SGD(training_data, 30, 7, 3.0, num_layers, biases, weights, 0.1, validation_data, test_data, True, True, True, True, True)