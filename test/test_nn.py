import pytest
import numpy as np
from final_project import utils

def test_convert_to_binary():
    # check that nucleotides are correctly written
    seq = ["ACGT"]
    bin_seq = utils.to_binary(seq)
    assert bin_seq == ["1000010000100001"]


def test_string_to_array():
    # verify that we can convert a list of strings to arrays
    binary_sequences = ["1000010000100001", "0100010000100001"]

    binary_array = utils.string_to_array(binary_sequences)

    verification_array = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
       [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]])

    assert np.array_equal(binary_array, verification_array)

