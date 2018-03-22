# generate a set of training examples

import os
import sys
import utils
from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np
import random


file_path = os.path.join(os.path.sep, 'Users', 'student', 'GitHub', 'bmi203_final')


def rev_complement(seqs):
    reverse_set = set()
    for item in seqs:
        tmp = Seq(item)
        reverse_set.add(tmp.reverse_complement)
    return reverse_set



def read_pos():
    # read in the positive example file
    filename = 'rap1-lieb-positives.txt'
    pos_seqs = []
    with open(os.path.join(file_path, filename)) as f:
        for line in f:
            seq = line.strip()
            pos_seqs.append(seq)
    return pos_seqs



def read_neg(pos):
    """
    reads in negative data set, removes sequences that match positive examples
    """
    pos_set = set(pos)
    # read in the negative example file
    filename = 'yeast-upstream-1k-negative.fa'

    records = SeqIO.parse(open(os.path.join(file_path, filename)), 'fasta')
    neg_seqs = []
    for item in records:
        neg_seqs.append(item.seq)

    pos_rev = rev_complement(pos_set)
    positives = pos_set.union(pos_rev)

    # remove any examples containing a positive example
    neg_set = set()
    for seq in neg_seqs:
        for i in range(len(seq) - 16):
            new_seq = seq[i:i + 17]
            if new_seq not in positives:
                neg_set.add(seq[i:i + 17])


    return list(neg_set)


def master_training_set(pos, neg):
    """
    outputs a file containing positive and negative data sets for training the nn
    """
    np.random.seed(100)  # for replicability
    master = set()

    for posseq in pos:
        tmp = '01' + posseq
        master.add(tmp)

    number_of_neg_examples = len(pos) * 100

    random.shuffle(neg)

    for negseq in neg[0:number_of_neg_examples]:
        tmp = '10' + negseq
        master.add(tmp)

    return master



def write_output(pos, neg, master):
    """
    writes output files
    training_pos = all positive data
    training_neg = all negative data
    training = master training set to be used
    """

    with open(os.path.join(file_path, "training_pos.txt"), 'w+') as outset:
        # write the sets
        outset.write('\n'.join(x for x in pos))

    with open(os.path.join(file_path, "training_neg.txt"), 'w+') as outset:
        # write the sets
        outset.write('\n'.join(x for x in neg))

    with open(os.path.join(file_path, "training.txt"), 'w+') as outset:
        outset.write('\n'.join(x for x in master))




def training_sets():

    random.seed(100)  # set random seed

    pos_seqs = read_pos()
    neg_seqs = read_neg(pos_seqs)

    pos_seqs = utils.to_binary(pos_seqs)
    neg_seqs = utils.to_binary(neg_seqs)

    master_seqs = master_training_set(pos_seqs, neg_seqs)

    write_output(pos_seqs, neg_seqs, master_seqs)


if __name__ == '__main__':
    training_sets()

