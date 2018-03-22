import numpy as np
import os


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
    """
    takes in list of binary strings, outputs numpy array
    """
    rows = len(binary_strings)
    cols = len(binary_strings[0])

    seq_array = np.zeros((rows, cols), dtype=int)

    for i, seq in enumerate(binary_strings):
        for j, char in enumerate(seq):
            seq_array[i, j] = int(char)

    return seq_array


def training_data(file_path, filename, class_size):
    """
    read in the training data
    """
    sequences = []
    classes  = []

    with open(os.path.join(file_path, filename)) as f:
        for line in f:
            l = line.strip()
            classes.append(l[:class_size])
            sequences.append(l[class_size:])

    return (string_to_array(sequences), string_to_array(classes))




