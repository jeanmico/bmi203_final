# implements a neural network

#TODO:
# output model
# run model on test data
# gradient descent with backpropagation

import sys
import os
import numpy as np
from scipy.special import expit
import sklearn.metrics as metrics


file_path = os.path.join(os.path.sep, 'Users', 'student', 'GitHub', 'bmi203_final')

# seqs = dataset
# y = class membership of each seq

def training_data():
    """
    read in the training data
    """
    sequences = []
    classes  = []
    seqname = 'training_sets.txt'
    classname = 'training_class.txt'

    with open(os.path.join(file_path, seqname)) as f:
        for line in f:
            sequences.append(line.strip())

    with open(os.path.join(file_path, classname)) as f:
        for line in f:
            lineval = line.strip().split('\t')
            classes.append(lineval)

    class_array = np.zeros((len(classes), 2))

    for i, class_member in enumerate(classes):
        class_array[i][0] = class_member[0]
        class_array[i][1] = class_member[1]

    return (string_to_array(sequences), class_array)


def string_to_array(binary_strings):
    rows = len(binary_strings)
    cols = len(binary_strings[0])

    seq_array = np.zeros((rows, cols), dtype=int)

    for i, seq in enumerate(binary_strings):
        for j, char in enumerate(seq):
            seq_array[i, j] = int(char)

    return seq_array


## GLOBAL PARAMETERS
seqs, y = training_data()

#seqs = np.identity(8)
#y = np.identity(8)

# model size
example_ct = len(seqs)
input_dim = len(seqs[0]) # (length, in binary, of strings: number of columns in seqs)
output_dim = 2

#input_dim = 8
#output_dim = 8

# gradient descent params
epsilon = 0.01  # learning rate for gradient descent


def activation(x):
    """
    sigmoid function
    """
    return expit(x)



def loss_compute(model):
    """
    determines performance of model
    """



def class_predict(model, seq):
    """
    classifies input as true binding site or not (1 or 0)
    """

    # forward propagation
    w1 = model['w1']
    w2 = model['w2']

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


def model_build(nodes_in_hidden, passes):
    """
    learns the parameters for the model
    input: number of nodes in hidden layer, # of passes through for grd descent
    output: model (list of parameters)
    """

    np.random.seed(100)
    #input_dim = 68
    #output_dim =8

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

        l1_delta = l1_error*activation(layer1)

        # update our weights
        w2 += layer1.T.dot(l2_delta)
        w1 += layer0.T.dot(l1_delta)

        if i%100 == 0:
            error = np.mean(np.abs(l2_error))
            print(error)
            errors.append(error)

    weights = dict()
    weights['w1'] = w1
    weights['w2'] = w2
    return weights, layer2  # this is our output layer!


def neural_net():
    # build and train the model
    hidden_nodes = 10
    passes = 100
    model, model_output = model_build(hidden_nodes, passes)


    #for i in seqs:
    #    model_output.append(class_predict(model, i))
    
    roc = roc_score(y, np.asarray(model_output, dtype=float))
    print(roc)

    with open(os.path.join(file_path, "tracking_rd.txt"), 'a') as output:
        output.write('\n' + str(hidden_nodes) + ' ' + str(passes) + ' ' + str(roc))

    teststr = ['10000100100000010100010000100001001001001000010001000001010001000010']
    print(string_to_array(teststr))
    print(string_to_array(teststr).shape)
    print(class_predict(model, string_to_array(teststr)))



if __name__ == "__main__":
    neural_net()
