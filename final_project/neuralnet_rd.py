# implements a neural network

#TODO
# really gorgeous comments!

import sys
import os
import numpy as np
from final_project import utils
from scipy.special import expit
import sklearn.metrics as metrics
import random
import math


file_path = os.path.join(os.path.sep, 'Users', 'student', 'GitHub', 'bmi203_final')

## GLOBAL PARAMETERS

# model size
output_dim = 2
hidden_nodes = 10
passes = 1000

# gradient descent params
epsilon = .1  # learning rate for gradient descent
lambda_reg = .1  #regularization term


def crossval_sets(pos_ind, neg_ind, seqs, classes):
    testsize = .75

    random.shuffle(pos_ind)
    random.shuffle(neg_ind)

    poslen = math.floor(testsize * len(pos_ind))
    neglen = math.floor(testsize * len(neg_ind))

    pos_train = pos_ind[:poslen]
    neg_train = neg_ind[:neglen]

    train_ind = pos_train + neg_train

    train_seq = []
    train_class = []
    test_seq = []
    test_class = []

    for i in range(len(seqs)):
        if i in train_ind:
            train_seq.append(seqs[i])
            train_class.append(classes[i])
        else:
            test_seq.append(seqs[i])
            test_class.append(classes[i])


    return np.asarray(train_seq), np.asarray(train_class), np.asarray(test_seq), np.asarray(test_class)


def crossval(seqs, y):
    """
    takes in a dataset
    returns two datasets: training and testing
    """

    # there has to be a better way to do this, but np.where is confusing
    full_index = set([x for x in range(len(y))])
    pos_index = set()

    for index, i in enumerate(y):
        if i[0] == 0:
            pos_index.add(index)

    print(len(pos_index))
    neg_index = full_index - pos_index

    full_index = list(full_index)
    neg_index = list(neg_index)
    pos_index = list(pos_index)

    return pos_index, neg_index


def activation(x, deriv=False):
    """
    sigmoid function
    """
    sigmoid_out = 1/(1 + np.exp(-x))
    if deriv:
        return sigmoid_out*(1-sigmoid_out)
    else:
        return sigmoid_out


def class_predict(model, test_seqs, roc=False):
    """
    classifies input as true binding site or not (1 or 0)
    """
    w1 = model['w1']
    w2 = model['w2']
    # forward propagation
    l1 = activation(np.dot(test_seqs, w1))
    l2 = activation(np.dot(l1, w2))

    if roc:
        return l2

    predictions = []
    # return output as a single number
    for output in l2[:,]:
        max_arg = np.argmax(l2)
        if max_arg == 0:
            predictions.append(1 - l2[max_arg])
        else:
            predictions.append(l2[max_arg])
    return predictions


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
        w2 += epsilon * (layer1.T.dot(l2_delta) + lambda_reg*w2)
        w1 += epsilon * (layer0.T.dot(l1_delta) + lambda_reg*w1)

        if i%100 == 0:
            error = np.mean(np.abs(l2_error))
            #print(error)
            errors.append(error)

    model['w1'] = w1
    model['w2'] = w2
    return model, layer2  # this is our output layer!


def neural_net():
    # build and train the model

    cross_validation = False

    seqs, y = utils.training_data(file_path, "training.txt", output_dim)

    if not cross_validation:
        #build model using training set
        model, model_output = model_build(seqs, y, hidden_nodes, passes)

        roc = roc_score(y, np.asarray(model_output, dtype=float))
        print(roc)


    else:

        #subset your data for cross-validation
        master_seqs, master_class = utils.training_data(file_path, "training.txt", output_dim)
        pos_ind, neg_ind = crossval(master_seqs, master_class)

        master_seqs.tolist()
        master_class.tolist()

        for j in range(25):
            seqs, y, seq_test, y_test = crossval_sets(pos_ind, neg_ind, master_seqs, master_class)
            model, model_output = model_build(seqs, y, hidden_nodes, passes)
            
            predictions = class_predict(model, seq_test, True)
            roc = roc_score(y_test, np.asarray(predictions, dtype=float))
            print(roc)

            with open(os.path.join(file_path, "crossval_roc.txt"), 'a') as output:
                output.write('\n' + str(j) + ' ' + str(roc))

