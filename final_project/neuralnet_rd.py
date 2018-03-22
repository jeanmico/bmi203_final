# implements a neural network

#TODO:
# gradient descent with backpropagation
# better training scheme
# cross-validation
# test scenarios
# fix derivatives
# WRITE-UP
# parameter testing: hidden neurons, epsilon, (optional reg_lambda)

import sys
import os
import numpy as np
import utils
from scipy.special import expit
import sklearn.metrics as metrics


file_path = os.path.join(os.path.sep, 'Users', 'student', 'GitHub', 'bmi203_final')
teststr = ['10000100100000010100010000100001001001001000010001000001010001000010']
# seqs = dataset
# y = class membership of each seq

## GLOBAL PARAMETERS

#seqs = np.identity(8)
#y = np.identity(8)

# model size
output_dim = 8
hidden_nodes = 10
passes = 100

#input_dim = 8
#output_dim = 8

# gradient descent params
epsilon = 0.01  # learning rate for gradient descent


def crossval():
    """
    takes in a dataset
    returns two datasets: training and testing
    """


def activation(x, deriv=False):
    """
    sigmoid function
    """
    sigmoid_out = 1/(1 + np.exp(-x))
    if deriv:
        return sigmoid_out*(1-sigmoid_out)
    else:
        return sigmoid_out


def activation_to_deriv():
    return 


def loss_compute(model):
    """
    determines performance of model
    """



def class_predict(model, seq):
    """
    classifies input as true binding site or not (1 or 0)
    """
    w1 = model['w1']
    w2 = model['w2']
    # forward propagation
    l1 = activation(np.dot(seq, w1))
    l2 = activation(np.dot(l1, w2))

    # return output as a single number
    max_arg = np.argmax(l2)
    if max_arg == 0:
        return (1 - l2[max_arg])
    else:
        return l2[max_arg]


def roc_score(true_scores, model_output):
    true_scores_flat = []
    for score in true_scores[:,]:
        true_scores_flat.append(np.argmax(score))

    model_output_flat = []
    for output in model_output[:,]:
        if np.argmax(output) == 0:
            model_output_flat.append(1 - output[0])
        else:
            model_output_flat.append(output[1])

    roc = metrics.roc_auc_score(true_scores_flat, model_output_flat)
    return roc

def backprop(net, seq, y):
    """

    """
    print('ok')


def model_build(seqs, y, nodes_in_hidden, passes):
    """
    learns the parameters for the model
    input: number of nodes in hidden layer, # of passes through for grd descent
    output: model (list of parameters)
    """

    np.random.seed(100)
    example_ct = len(seqs)
    input_dim = len(seqs[0])

    w1 = 2*np.random.randn(input_dim, nodes_in_hidden) - 1 # from layer0 to hidden
    w2 = 2*np.random.randn(nodes_in_hidden, output_dim) - 1 # from hidden to output
    model = dict()  # dictionary to hold the parameters

    errors = []

    # in each pass, we try to learn our weight parameters better
    for i in range(passes):

        layer0 = seqs  # input layer; training data
        layer1 = activation(np.dot(layer0, w1))  # hidden layer
        layer2 = activation(np.dot(layer1, w2))  # output layer: our prediction


        l2_error = y - layer2
        l2_delta = l2_error*activation(layer2)
        l1_error = l2_delta.dot(w2.T)  # this is the backpropagation step!!
        l1_delta = l1_error*activation(layer1, True)

        # update our weights
        w2 += layer1.T.dot(l2_delta)
        w1 += layer0.T.dot(l1_delta)

        if i%1000 == 0:
            error = np.mean(np.abs(l2_error))
            print(error)
            errors.append(error)

    model['w1'] = w1
    model['w2'] = w2
    return model, layer2  # this is our output layer!


def neural_net():
    # build and train the model

    cross_validation = False

    seqs, y = utils.training_data(file_path, "838.txt", 8)
    print(seqs)
    print(y)

    if not cross_validation:
        #build model using training set
        model, model_output = model_build(seqs, y, hidden_nodes, passes)
        print(model_output)

    else:
        #subset your data for cross-validation
        for j in range(20):
            seqs, y = crossval()
            model, model_output = model_build(seqs, y, hidden_nodes, passes)
        
            roc = roc_score(y, np.asarray(model_output, dtype=float))
            print(roc)

            with open(os.path.join(file_path, "tracking_cv.txt"), 'a') as output:
                output.write('\n' + str(hidden_nodes) + ' ' + str(passes) + ' ' + str(roc))



if __name__ == "__main__":
    neural_net()
