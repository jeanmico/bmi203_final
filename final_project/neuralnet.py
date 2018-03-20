# implements a neural network


import os
import numpy as np
from scipy.special import expit

# seqs = dataset
# y = class membership of each seq

def training_data():
    """
    read in the training data
    """
    print('oh wow')

def to_binary(seqs):
    """
    takes in list of sequences, returns list of sequences in binary
    """
    bin_dict = dict()
    bin_dict["A"] = "1000"
    bin_dict["C"] = "0100"
    bin_dict["G"] = "0010"
    bin_dict["T"] = "0001"

    bin_seqs = []
    for seq in seqs:
        bin_seq = ""
        for char in seq:
            bin_seq += bin_dict[char]
        bin_seqs.append(bin_seq)

    return bin_seqs

def string_to_array(binary_strings):
    rows = len(binary_strings)
    cols = len(binary_strings[0])

    seq_array = np.zeros((rows, cols), dtype=int)

    for i, seq in enumerate(binary_strings):
        for j, char in enumerate(seq):
            seq_array[i, j] = int(char)
    print(seq_array)
    return seq_array


## GLOBAL PARAMETERS
test = ['0001', '0001', '0010']
seqs = string_to_array(test)
y = [1, 0, 0]

# model size
input_dim = 4 # (length, in binary, of strings: number of columns in seqs)
output_dim = 2
example_ct = len(seqs) 

# gradient descent params
epsilon = 0.01
lambda_reg = 0.01


def model_math(model):
    w1, b1, w2, b2 = model['w1'], model['b1'], model['w2'], model['b2']

    #forward propagation

    z1 = x.dot(w1) + b1
    a1 = expit(z1)  # sigmoid function
    z2 = a1.dot(w2) + b2
    exp_scores= np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def loss_compute(model):
    """
    determines performance of model
    """
    w1, b1, w2, b2 = model['w1'], model['b1'], model['w2'], model['b2']
    z1 = seq.dot(w1) + b1
    a1 = expit(z1)  # sigmoid function
    z2 = a1.dot(w2) + b2
    exp_scores= np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # calculating the loss
    data_loss = np.sum(-1*np.log(procs[range(example_ct), y]))

    # adding the regularization term (optional)
    data_loss += lambda_reg/2 * (np.sum(np.square(w1))) + np.sum(np.square(w2))
    return 1./example_ct * data_loss


def class_predict(model, seq):
    """
    classifies input as true binding site or not (1 or 0)
    """

    w1, b1, w2, b2 = model['w1'], model['b1'], model['w2'], model['b2']

    #forward propagation

    z1 = seq.dot(w1) + b1
    a1 = expit(z1)  # sigmoid function
    z2 = a1.dot(w2) + b2
    exp_scores= np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


def model_build(nodes_in_hidden, passes):
    """
    learns the parameters for the model
    input: number of nodes in hidden layer, # of passes through for grd descent
    output: model (list of parameters)
    """
    

    np.random.seed(100)
    w1 = np.random.randn(input_dim, nodes_in_hidden) / np.sqrt(input_dim)
    b1 = np.zeros((1, nodes_in_hidden))
    w2 = np.random.randn(nodes_in_hidden, output_dim) / np.sqrt(nodes_in_hidden)
    b2 = np.zeros((1, output_dim))

    model = dict()  # dictionary to hold the parameters

    for i in range(0, passes):
        z1 = seqs.dot(w1) + b1
        a1 = expit(z1)
        z2 = a1.dot(w2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # backpropagation
        delta3 = probs
        delta3[range(example_ct), y] -= 1
        dw2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(w2.T) * (1 - np.power(a1, 2))
        dw1 =  np.dot(seqs.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # regularization terms

        dw2 += lambda_reg * w2

        dw1 = lambda_reg * w1

        # assign new parameters
        model['w1'] = w1
        model['w2'] = w2
        model['b1'] = b1
        model['b2'] = b2

    return model


def neural_net():
    # build and train the model
    print(model_build(3, 20))



if __name__ == "__main__":
    neural_net()
