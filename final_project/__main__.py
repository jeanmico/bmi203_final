import sys
from .training_sets import training_sets
from .neuralnet_rd import neural_net

arguments = sys.argv[1:]

if arguments[0] == "True":
    training_sets()
    
neural_net()
