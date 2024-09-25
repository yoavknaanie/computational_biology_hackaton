import numpy as np
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from Bio.Phylo.TreeConstruction import DistanceMatrix
from typing import List

def turn_matrix_to_triangle(mat: np.ndarray):
    """
    Convert a square matrix into a lower triangular matrix.

    Parameters:
    - mat (np.ndarray): Input square matrix.

    Returns:
    - List[List[float]]: Lower triangular matrix represented as a list of lists.
    """
    return [ls[:i + 1].tolist() for i, ls in enumerate(mat)]

def create_tree(mat: np.ndarray, bacteria_names: List[str]):
    """
    Construct a phylogenetic tree using neighbor-joining algorithm.

    Parameters:
    - mat (np.ndarray): Distance matrix containing evolutionary distances between bacteria.
    - bacteria_names (List[str]): List of bacteria names corresponding to the rows/columns of the distance matrix.

    Returns:
    - Bio.Phylo.BaseTree.Tree: Constructed phylogenetic tree.
    """
    mat = turn_matrix_to_triangle(mat)
    # Convert the distance matrix to a DistanceMatrix object
    dm = DistanceMatrix(bacteria_names, mat)

    # Use the DistanceTreeConstructor to build a tree
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(dm)

    return tree
