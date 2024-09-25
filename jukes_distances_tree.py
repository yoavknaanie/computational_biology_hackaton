import numpy as np
from Bio import Phylo
import neighbors_joining as nj

def build_tree_from_continues_time(time_dict, bacteria_names):
    dist_mat = init_matrix(time_dict, bacteria_names)
    tree = nj.create_tree(dist_mat, bacteria_names)

    # Step 3: Visualize or save the tree (in this example, visualize using ASCII)
    Phylo.draw_ascii(tree)
    return tree

def init_matrix(time_dict, bacteria_names):
    size = len(bacteria_names)
    mat = np.zeros([size, size])
    for i in range(size):
        for j in range(size):
            if j > i:
                mat[i][j] = time_dict[(bacteria_names[i], bacteria_names[j])]
                mat[j][i] = time_dict[(bacteria_names[i], bacteria_names[j])]

    return mat
