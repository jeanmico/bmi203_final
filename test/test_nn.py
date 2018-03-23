import pytest
from final_project import utils

def test_convert_to_binary():
    # check that nucleotides are correctly written
    seq = "ACGT"
    bin_seq = utils.to_binary(seq)

    assert bin_seq == "1000010000100001"