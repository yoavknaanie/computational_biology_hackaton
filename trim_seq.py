import argparse
import dist_matrix as dm
import sys
import os
import time
import neighbors_joining
import numpy as np


def align_2_sequences(bac_a, bac_b, score_matrix):
    seq_a = ""
    seq_b = ""
    for header, sequence in dm.fastaread(bac_a):
        seq_a += sequence
    for header, sequence in dm.fastaread(bac_b):
        seq_b += sequence
    return dm.overlap_align(seq_a, seq_b, score_matrix)
    #return dm.local_align(seq_a, seq_b, score_matrix)

if __name__ == '__main__':
    short_seq= "16s\\E_coli.fna"
    long_seq= "16s\\Bacillus_subtilis.fna"
    score_mat = dm.matrix_read("score_matrix.tsv")
    align_2_sequences(short_seq, long_seq, score_mat)

