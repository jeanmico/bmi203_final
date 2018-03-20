# generate a set of training examples

import os
import sys
from Bio import SeqIO


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
    pos_set = set(pos)
    # read in the negative example file
    filename = 'yeast-upstream-1k-negative.fa'

    records = SeqIO.parse(open(os.path.join(file_path, filename)), 'fasta')
    neg_seqs = []
    for item in records:
        neg_seqs.append(item.seq)

    # remove any examples containing a positive example
    neg_set = set()
    for seq in neg_seqs:
        for i in range(len(seq) - 16):
            new_seq = seq[i:i + 17]
            if new_seq not in pos_set:
                neg_set.add(seq[i:i + 17])
        if len(neg_set) > 100000:  # remove this if we want everything: 2982679
            break

    return neg_seqs



def write_output(pos, neg):
    #write output files
    num_pos = [1]*len(pos)
    num_neg = [0]*len(neg)
    classes = num_pos + num_neg

    with open(os.path.join(file_path, "training_sets.txt"), 'w+') as outset:
        # write the sets
        print('ok')

    with open(os.path.join(file_path, "training_class.txt"), 'w+') as outclass:
        # write the class membership
        outclass.write('\n'.join(str(x) for x in classes))




def training_sets():

    pos_seqs = read_pos()
    neg_seqs = read_neg(pos_seqs)

    pos_seqs = to_binary(pos_seqs)
    neg_seqs = to_binary(neg_seqs)


    write_output(pos_seqs, neg_seqs)


if __name__ == '__main__':
    training_sets()

