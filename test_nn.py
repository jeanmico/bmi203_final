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
    assert 1 == 1