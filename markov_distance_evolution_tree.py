import math
from typing import Dict, Tuple, List
import numpy as np
from scipy.linalg import expm

WINDOW_SIZE = 11
JC_MATRIX = np.array([[-3, 1, 1, 1],
                      [1, -3, 1, 1],
                      [1, 1, -3, 1],
                      [1, 1, 1, -3]])
nt2int_MAT = {'A': 0, 'G': 1, 'C': 2, 'T': 3}
MIN_T = 1
MAX_T = 1000
JUMP = 1
ALPHA = 0.1

def parse_phylip(filename):
    """
    Parse a PHYLIP format file and return sequences and sequence length.

    Args:
        filename (str): The path to the PHYLIP format file.

    Returns:
        dict: A dictionary containing sequences with their respective identifiers as keys.
        int: The length of the sequences.
    """
    with open(filename) as f:
        lines = f.readlines()
    num_seqs, seq_len = map(int, lines[0].split())
    sequences = {}
    for line in lines[1:]:
        parts = line.strip().split()
        sequences[parts[0]] = parts[1]
    return sequences, seq_len

def get_sequence_windows(sequences, window_size=WINDOW_SIZE):
    """
    Generate sequence windows of specified size from input sequences.

    Args:
        sequences (dict): A dictionary containing sequences with their respective identifiers as keys.
        window_size (int, optional): The size of the sliding window. Defaults to 11.

    Returns:
        dict: A dictionary containing sequence windows with their respective identifiers as keys.
    """
    windows = {key: [sequences[key][i:i + window_size] for i in
                     range(len(sequences[key]) - window_size + 1)]
               for key in sequences}
    return windows

def transition_matrix(time):
    """
    Calculate the transition matrix given a "time" parameter.

    Args:
        time (float): The time parameter.

    Returns:
        numpy.ndarray: The transition matrix.
    """
    return expm(time * JC_MATRIX)

def find_all_pairs(sequences):
    """
    Find all possible sequence pairs.

    Args:
        sequences (dict): A dictionary containing sequences with their respective identifiers as keys.

    Returns:
        list: A list of tuples representing all possible sequence pairs.
    """
    num_of_sequences = len(sequences)
    seq_list = list(sequences.keys())
    pairs = []
    for i in range(num_of_sequences):
        for j in range(num_of_sequences):
            if j > i:
                a = seq_list[i]
                b = seq_list[j]
                tup = (a, b)
                pairs.append(tup)
    return pairs

def nt2int(seq):
    """
    Convert nucleotide sequence to integers.

    Args:
        seq (str): The nucleotide sequence.

    Returns:
        list: A list of integers representing the sequence.
    """
    indices = []
    for i in seq:
        indices.append(nt2int_MAT[i])
    return indices

def likelihood(seq_a, seq_b, t) -> float:
    """
    Calculate the likelihood given couples, actual sequences, and t.

    Returns:
        float: The likelihood value.
    """
    couples_likely = 0
    jc_matrix = transition_matrix(ALPHA * t)
    indices_seq_a = nt2int(seq_a)
    indices_seq_b = nt2int(seq_b)
    for i in range(len(seq_b)):
        jc_val = jc_matrix[indices_seq_a[i], indices_seq_b[i]]
        if jc_val != 0:
            couples_likely += math.log2(jc_val)
        else:
            couples_likely += -np.inf
    return couples_likely

def seqs_to_windows(sequences, windows_num):
    """
    Convert sequences to windows for likelihood calculation.

    Args:
        sequences (dict): A list of sequence identifiers.
        windows_num (int): The number of windows.

    Returns:
        list: A list of dictionaries containing sequences for each window.
    """
    windows = get_sequence_windows(sequences)
    windows_seqs = []
    for idx in range(windows_num):
        curr_dict = {}
        for seq in sequences.keys():
            curr_dict[seq] = windows[seq][idx]
        windows_seqs.append(curr_dict)
    return windows_seqs

def find_best_t(couple, number_of_windows, seqs_by_window) -> float:
    """
    Find the best alpha for given couples and sequences.

    Returns:
        float: best T for the couple
    """
    ts = np.arange(MIN_T, MAX_T + JUMP, JUMP)
    windows_t_array = np.zeros(number_of_windows)
    for i in range(number_of_windows):
        max_likely = -np.inf
        window_chosen_t = None
        for t in ts:
            first_seq = seqs_by_window[i][couple[0]]
            second_seq = seqs_by_window[i][couple[1]]
            ll = likelihood(first_seq, second_seq, t)
            if ll >= max_likely:
                window_chosen_t = t
                max_likely = ll
        windows_t_array[i] = window_chosen_t
    return np.mean(windows_t_array)

def find_evolution_distance(sequences, min_length):
    """
    Find evolutionary distances between sequence pairs.

    Args:
        sequences (dict): A dictionary containing sequences with their respective identifiers as keys.
        min_length (int): The minimum length among all sequences.

    Returns:
        dict: A dictionary containing evolutionary distances between sequence pairs.
    """
    couples = find_all_pairs(sequences)
    num_of_windows = min_length - (WINDOW_SIZE - 1)
    seqs_by_windows = seqs_to_windows(sequences, num_of_windows)
    best_ts = {}
    for couple in couples:
        best_t = find_best_t(couple, num_of_windows, seqs_by_windows)
        best_ts[couple] = np.round(best_t, 2)
    return best_ts
