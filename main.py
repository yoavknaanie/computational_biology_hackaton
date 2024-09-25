import argparse
import dist_matrix
import os
import neighbors_joining as nj
import numpy as np
import estimate_tree_lengths as etl
from Bio import Phylo
import matplotlib.pyplot as plt
import markov_distance_evolution_tree

def read_fasta_seq(file_path: str) -> str:
    """
    Read sequences from a FASTA file.

    Args:
        file_path (str): Path to the FASTA file.

    Returns:
        str: Concatenated sequence read from the file.
    """
    with open(file_path) as file:
        sequence = ""
        for line in file:
            line = line.strip()
            if not line.startswith(">"):  # Skip header lines
                sequence += line
    return sequence

def parser():
    """
    Parse command-line arguments and read sequences from files.

    Returns:
        tuple: A tuple containing bac_list (list of file paths),
               score_matrix (numpy array), bacteria_names (list of strings),
               dict_name_seq (dictionary mapping bacteria names to sequences),
               min_length (minimum length of sequences).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", type=str, help="path to the sequences")
    parser.add_argument("score_matrix", type=str, help="score matrix")
    command_args = parser.parse_args()
    folder_path = command_args.folder_path
    bac_names = os.listdir(folder_path)
    bac_list = []
    bacteria_names = []
    dict_name_seq = dict()
    for bac in bac_names:
        file_name = rf"{folder_path}\{bac}"
        bac_list.append(file_name)
        bacteria_name = bac.split(".")[0]
        bacteria_names.append(bacteria_name)
        seq = read_fasta_seq(file_name)
        dict_name_seq[bacteria_name] = seq
    score_matrix = dist_matrix.matrix_read(command_args.score_matrix)
    min_length = min(len(value) for value in dict_name_seq.values())
    return bac_list, score_matrix, bacteria_names, dict_name_seq, min_length

def alignment_based_tree (bac_files_names, score_matrix, bacteria_names, time_dict):
    """
    Generate a phylogenetic tree based on sequence alignment.

    Args:
        bac_files_names (list): List of file paths containing sequences.
        score_matrix (numpy array): Scoring matrix for sequence alignment.
        bacteria_names (list): Names of bacteria.
        time_dict (dict): Dictionary containing evolution distances between bacteria.

    Returns:
        Bio.Phylo.BaseTree.Tree: Phylogenetic tree based on sequence alignment.
    """
    distances = dist_matrix.create_matrix(bac_files_names, score_matrix)
    tree = nj.create_tree(distances, bacteria_names) #neighbors joining
    Phylo.draw(tree)
    return etl.process_tree_creation(tree, bacteria_names, time_dict)

def jk_markov_based_tree(time_dict, bacteria_names):
    """
    Generate a phylogenetic tree based on continuous Markov evolution model.

    Args:
        time_dict (dict): Dictionary containing evolution distances between bacteria.
        bacteria_names (list): Names of bacteria.

    Returns:
        Bio.Phylo.BaseTree.Tree: Phylogenetic tree based on continuous Markov model.
    """
    size = len(bacteria_names)
    dist_mat = np.zeros([size, size])
    for i in range(size):
        for j in range(size):
            if j > i:
                dist_mat[i][j] = time_dict[(bacteria_names[i], bacteria_names[j])]
                dist_mat[j][i] = time_dict[(bacteria_names[i], bacteria_names[j])]

    tree = nj.create_tree(dist_mat, bacteria_names) #neighbors joining
    return etl.process_tree_creation(tree, bacteria_names, time_dict)

def rename_inner_nodes(node):
    """
    Recursively traverses the tree and renames inner nodes whose names start with "inner".

    Parameters:
    - node (Phylo.BaseTree.Clade): A node in the tree.

    Returns:
    - None
    """
    if node.name and node.name.startswith("Inner"):
        node.name = ""  # Set the name to an empty string for inner nodes
    if node.clades:
        for child in node.clades:
            rename_inner_nodes(child)

def draw_tree(tree, title):
    """
    Draw and save a phylogenetic tree.

    Args:
        tree (Bio.Phylo.BaseTree.Tree): Phylogenetic tree to draw.
        title (str): Title of the tree.
    """
    rename_inner_nodes(tree.clade)
    fig = plt.figure(figsize=(10, 6), dpi=300)
    plt.rcParams['font.size'] = 9
    axes = fig.add_subplot(1, 1, 1)
    plt.rcParams["lines.linewidth"] = 0.8  # Adjust line width as needed
    Phylo.draw(tree,
               axes=axes,
               do_show=False,
               branch_labels=lambda c: f'{round(c.branch_length, 2):.2f}' if c.branch_length else '')
    axes.set_title(f'DNA Polymerase III Phylogenetic Tree according to {title}', fontsize=10)  # Add your title here
    plt.savefig(f'phylogenetic_tree_{title}.png')

if __name__ == '__main__':
    bac_files_names, score_matrix, bacteria_names, dict_name_seq, min_length = parser()
    time_dict = markov_distance_evolution_tree.find_evolution_distance(dict_name_seq, min_length)
    print(time_dict)
    # time_dict_16s = {('Bacillus subtilis', 'Clostridium perfringens'): 283.49,
    #              ('Bacillus subtilis', 'Escherichia albertii'): 395.11,
    #              ('Bacillus subtilis', 'Escherichia coli'): 338.28,
    #              ('Bacillus subtilis', 'Klebsiella pneumoniae'): 383.1,
    #              ('Bacillus subtilis', 'Streptococcus pneumoniae'): 376.03,
    #              ('Clostridium perfringens', 'Escherichia albertii'): 442.38,
    #              ('Clostridium perfringens', 'Escherichia coli'): 487.5,
    #              ('Clostridium perfringens', 'Klebsiella pneumoniae'): 486.16,
    #              ('Clostridium perfringens', 'Streptococcus pneumoniae'): 290.23,
    #              ('Escherichia albertii', 'Escherichia coli'): 1.02,
    #              ('Escherichia albertii', 'Klebsiella pneumoniae'): 5.34,
    #              ('Escherichia albertii', 'Streptococcus pneumoniae'): 340.73,
    #              ('Escherichia coli', 'Klebsiella pneumoniae'): 6.62,
    #              ('Escherichia coli', 'Streptococcus pneumoniae'): 375.34,
    #              ('Klebsiella pneumoniae', 'Streptococcus pneumoniae'): 355.65}
    tree_align= alignment_based_tree(bac_files_names, score_matrix, bacteria_names, time_dict)
    tree_markov= jk_markov_based_tree(time_dict, bacteria_names)
    draw_tree(tree_align, "Sequences Alignment")
    draw_tree(tree_markov, "Continuous Markov")
