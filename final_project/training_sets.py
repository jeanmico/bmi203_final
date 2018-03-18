# generate a set of training examples

import os
import sys


file_path = os.path.join(os.path.sep, 'Users', 'student', 'GitHub', 'bmi203_final')

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

def read_pos()
    # read in the positive example file
    filename = 'rap1-lieb-positives.txt'




def read_neg()
    # read in the negative example file
    filename = 'yeast-upstream-1k-negative.fa'
    


def write_output()

