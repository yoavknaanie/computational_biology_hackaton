import numpy as np
from itertools import groupby

A = 0
C = 1
G = 2
T = 3
GAP = 4
LINE = [A, C, G, T, GAP]
LETTERS = {"A": A, "C": C, "G": G, "T": T}
LEFT = 0
UP = 1
DIAGONAL = 2
SEQ_START = 3

def create_matrix(sequences_path, score_matrix):
    """
    Create a distance matrix based on global sequence alignment.

    Args:
        sequences_path (list): List of file paths containing sequences.
        score_matrix (list): Scoring matrix for sequence alignment.

    Returns:
        numpy.ndarray: Distance matrix calculated based on sequence alignment.
    """
    number_of_bacteria = len(sequences_path)
    distance_matrix = np.zeros((number_of_bacteria, number_of_bacteria))
    sequences = []
    for i in range(number_of_bacteria):
        seq = ''
        for header, sequence in fastaread(sequences_path[i]):
            seq += sequence
        sequences.append(seq)

    # Calculate the shortest sequence length
    min_length = min(len(seq) for seq in sequences)

    # Update sequences to match the shortest length
    sequences = [seq[:min_length] for seq in sequences]

    for i in range(number_of_bacteria):
        for j in range(number_of_bacteria):
            if j >= i:
                score = global_align(sequences[i], sequences[j], score_matrix)
                distance_matrix[i][j] = score
                distance_matrix[j][i] = score

    distance_matrix = fix_mat(distance_matrix)
    return distance_matrix

def fix_mat(mat):
    """
    Fix the distance matrix by adjusting scores.

    Args:
        mat (numpy.ndarray): Distance matrix to fix.

    Returns:
        numpy.ndarray: Fixed distance matrix.
    """
    min_score = np.min(mat)
    mat += np.abs(min_score)
    max_score = np.max(mat)
    mat = max_score - mat
    return mat

def fastaread(fasta_name):
    """
    Read a fasta file. For each sequence in the file, yield the header and the actual sequence.
    """
    f = open(fasta_name)
    faiter = (x[1] for x in groupby(f, lambda line: line.startswith(">")))
    for header in faiter:
        header = next(header)[1:].strip()
        seq = "".join(s.strip() for s in next(faiter))
        yield header, seq

def matrix_read(mat_name):
    """
    Read a scoring matrix from file.

    Args:
        mat_name (str): File path to the scoring matrix.

    Returns:
        list: Scoring matrix.
    """
    score = []
    f = open(mat_name)
    f.readline()
    for i in range(5):
        row = [int(x) for x in f.readline().split("\t")[1:]]
        score.append(row)
    return score

def global_align(seq_a, seq_b, score_mat):
    """
    Perform global sequence alignment and return the alignment score.

    Args:
        seq_a (str): First sequence for alignment.
        seq_b (str): Second sequence for alignment.
        score_mat (list): Scoring matrix for alignment.

    Returns:
        int: Alignment score.
    """
    L = [[0 for _ in range(len(seq_b)+1)] for _ in range(len(seq_a)+1)]  # seq_a is rows seq_b col
    L_moves = [[0 for _ in range(len(seq_b)+1)] for _ in range(len(seq_a)+1)]
    ali_a = ""
    ali_b = ""

    # init first col
    for i, letter in enumerate(seq_a):
        L[i+1][0] = L[i][0] + score_mat[LETTERS[letter]][GAP]
        L_moves[i+1][0] = UP
    # init first row
    for j, letter in enumerate(seq_b):
        L[0][j+1] = L[0][j] + score_mat[LETTERS[letter]][GAP]
        L_moves[0][j + 1] = LEFT

    for j in range(1, len(seq_b)+1):
        for i in range(1, len(seq_a)+1):
            left_step = L[i][j-1] + score_mat[LETTERS[seq_b[j-1]]][GAP]
            up_step = L[i-1][j] + score_mat[LETTERS[seq_a[i-1]]][GAP]
            dig_step = L[i-1][j-1] + score_mat[LETTERS[seq_a[i-1]]][LETTERS[seq_b[j-1]]]
            directions = np.array([left_step, up_step, dig_step])
            L_moves[i][j] = np.argmax(directions)
            L[i][j] = max(left_step, up_step, dig_step)

    # writing alignment
    i = len(seq_a)
    j = len(seq_b)
    while not(i == 0 and j == 0):
        if L_moves[i][j] == LEFT:
            j -= 1
            ali_a = "-" + ali_a
            ali_b = seq_b[j] + ali_b
        if L_moves[i][j] == UP:
            i -= 1
            ali_a = seq_a[i] + ali_a
            ali_b = "-" + ali_b
        if L_moves[i][j] == DIAGONAL:
            i -= 1
            j -= 1
            ali_a = seq_a[i] + ali_a
            ali_b = seq_b[j] + ali_b

    score = L[len(seq_a)][len(seq_b)]
    return score
