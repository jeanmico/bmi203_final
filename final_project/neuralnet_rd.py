# implements a neural network


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
            classes.append(line.strip())

    return (string_to_array(sequences), np.asarray(classes, dtype=int))


def string_to_array(binary_strings):
    rows = len(binary_strings)
    cols = len(binary_strings[0])

    seq_array = np.zeros((rows, cols), dtype=int)

    for i, seq in enumerate(binary_strings):
        for j, char in enumerate(seq):
            seq_array[i, j] = int(char)

    return seq_array


## GLOBAL PARAMETERS

#seqs, y = training_data()



# model size
#input_dim = 68 # (length, in binary, of strings: number of columns in seqs)
#output_dim = 2
#example_ct = len(seqs) 

seqs = np.identity(8)
y = np.identity(8)
print(y.shape)
print(type(y))

example_ct = len(seqs) 

input_dim = 8
output_dim = 8

# gradient descent params
epsilon = 0.01
lambda_reg = 0.01


def activation(x):

    return expit(x)



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
    #return np.argmax(probs, axis=1)
    #return np.argmax(probs)
    if np.argmax(probs)==0:
        return 1 - np.max(probs)
    else:
        return np.max(probs)

def roc_score(true_scores, model_output):
    roc = metrics.roc_auc_score(true_scores, model_output)
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
    input_dim =8
    output_dim =8

    w1 = 2*np.random.randn(input_dim, nodes_in_hidden) - 1
    w2 = 2*np.random.randn(nodes_in_hidden, output_dim) - 1

    print(w1.shape)
    print(w2.shape)

    model = dict()  # dictionary to hold the parameters

    for i in range(passes):
        b1 = 0
        b2 = 0


        layer0 = seqs
        layer1 = activation(np.dot(layer0, w1))
        layer2 = activation(np.dot(layer1, w2))

        print('l1: ' + str(layer1.shape))
        print(layer2.shape)

        l2_error = y - layer2

        l2_delta = l2_error*activation(layer2)
        print('l2delt ' + str(l2_delta.shape))

        l1_error = l2_delta.dot(w2.T)  # this is the backpropagation step!!

        l1_delta = l1_error*activation(layer1)

        # update!
        w2 += layer1.T.dot(l2_delta)
        w1 += layer0.dot(l1_delta)


    print(layer2)


def neural_net():
    # build and train the model
    hidden_nodes = 3
    passes = 120000
    model = model_build(hidden_nodes, passes)

    model_output = []
    for i in seqs:
        model_output.append(class_predict(model, i))
    
    roc = roc_score(y, np.asarray(model_output, dtype=float))
    print(roc)

    with open(os.path.join(file_path, "tracking.txt"), 'a') as output:
        output.write('\n' + str(hidden_nodes) + ' ' + str(passes) + ' ' + str(roc))




if __name__ == "__main__":
    neural_net()
