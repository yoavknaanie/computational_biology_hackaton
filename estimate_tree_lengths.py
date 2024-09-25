import numpy as np
from Bio.Phylo.BaseTree import Tree
import scipy.optimize as opt

def rebuild_phylo_tree_from_clade_pairs(shared_edge_pairs, lengths):
    """
    Rebuild a phylogenetic tree from shared edge pairs and corresponding branch lengths.

    Args:
        shared_edge_pairs (list): List of tuples representing shared edge pairs.
        lengths (list): List of branch lengths.

    Returns:
        Bio.Phylo.BaseTree.Tree: Rebuilt phylogenetic tree.
    """
    # Track all clades to find the root later
    all_clades = set()
    child_clades = set()

    for (parent_clade, child_clade), length in zip(shared_edge_pairs, lengths):
        parent_clade.clades.append(child_clade)
        child_clade.branch_length = length

        # Keep track of all clades and which clades are children
        all_clades.add(parent_clade)
        child_clades.add(child_clade)

    # The root clades are those that never appear as a child
    root_clades = all_clades - child_clades

    # Assuming there's a single root in the tree
    if root_clades:
        root_clade = root_clades.pop()
        return Tree(root=root_clade)
    else:
        return None

def get_order(edges):
    """
    Get the order of vertices in the edge list.

    Args:
        edges (list): List of edge tuples.

    Returns:
        list: List of vertex names in the order they appear in the edge list.
    """
    ordered = []
    for e in edges:
        child = e[1]
        ordered.append(child.name)
    return ordered

def create_trajectory_matrix(tree, sequences, edges):
    """
    Create a trajectory matrix representing the transitions between sequences along the tree.

    Args:
        tree (Bio.Phylo.BaseTree.Tree): Phylogenetic tree.
        sequences (list): List of sequence names.
        edges (list): List of edge tuples.

    Returns:
        numpy.ndarray: Trajectory matrix.
        list: List of sequence pairs.
    """
    # Create an empty matrix
    matrix = np.zeros((len(sequences) * (len(sequences) - 1) // 2, len(edges)), dtype=int)

    pairs_list = []  # list of pairs of sequences in the same order as the rows in the matrix
    vertices_ordered = get_order(edges)

    # Populate the matrix
    index = 0
    for i in range(len(sequences)):
        for j in range(i):
            seq1, seq2 = sequences[i], sequences[j]
            pairs_list.append((seq1, seq2))
            trace = tree.trace(seq1, seq2)
            trace_leaves = [node.name for node in trace]
            leaves = set([seq1, seq2] + trace_leaves)
            common_ancestor = tree.common_ancestor(seq1, seq2).name
            leaves.remove(common_ancestor)
            for e in edges:
                child = e[1]
                if child.name in leaves:
                    matrix[index, vertices_ordered.index(child.name)] = 1
            index += 1
    return matrix, pairs_list

def calculate_non_neg_ls(trajectory_matrix, pairs_list, time_dict):
    """
    Calculate the linear regression of the trajectory matrix and the distances matrix.

    Args:
        trajectory_matrix (numpy.ndarray): Trajectory matrix.
        pairs_list (list): List of sequence pairs.
        time_dict (dict): Dictionary containing evolutionary distances between sequence pairs.

    Returns:
        numpy.ndarray: Linear regression results.
    """
    vector_time = np.zeros((trajectory_matrix.shape[0]), dtype=float)
    for pair in pairs_list:
        if pair in time_dict:
            vector_time[pairs_list.index(pair)] = time_dict[pair]
        elif (pair[1], pair[0]) in time_dict:
            vector_time[pairs_list.index(pair)] = time_dict[(pair[1], pair[0])]
    reg = opt.nnls(trajectory_matrix, vector_time)
    return reg[0]

def process_tree_creation(tree, bacteria_names, time_dict):
    """
    Process tree creation based on trajectory matrix and evolutionary distances.

    Args:
        tree (Bio.Phylo.BaseTree.Tree): Phylogenetic tree.
        bacteria_names (list): List of bacterial names.
        time_dict (dict): Dictionary containing evolutionary distances between sequence pairs.

    Returns:
        Bio.Phylo.BaseTree.Tree: Processed phylogenetic tree.
    """
    edges = find_shared_edge_pairs(tree)
    trajectory_matrix, pairs_list = create_trajectory_matrix(tree, bacteria_names, edges)
    vector_ls_result = calculate_non_neg_ls(trajectory_matrix, pairs_list, time_dict)
    return rebuild_phylo_tree_from_clade_pairs(edges, vector_ls_result)

def find_shared_edge_pairs(tree):
    """
    Find shared edge pairs in a phylogenetic tree.

    Args:
        tree (Bio.Phylo.BaseTree.Tree): The phylogenetic tree.

    Returns:
        list: A list of tuples representing shared edge pairs.
    """
    shared_edge_pairs = []
    for node in tree.find_clades():
        if not node.is_terminal():
            children = node.clades
            for child in children:
                shared_edge_pairs.append((node, child))
    return shared_edge_pairs
