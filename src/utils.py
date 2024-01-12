import torch
import numpy as np
from scipy.sparse import coo_matrix
import pickle
from random import shuffle
from copy import deepcopy
import scipy
'''
Some of the utility functions are taken from the following repository: https://github.com/ggoh29/Simplicial-neural-network-benchmark
'''

def tensor_to_sparse(tensor_dict):
    "Converts a 3 dim dict matrix to a sparse matrix"
    return torch.sparse_coo_tensor(indices = tensor_dict["indices"], values = tensor_dict["values"].squeeze(),size = tensor_dict["shape"])

def sparse_to_tensor(matrix):
    '''Converts a sparse matrix to a 3 x N matrix. This function is used in combination with the function to_sparse_coo.
    The reason is that pytorch dataloaders cannot handle the sparse format'''

    indices = matrix.coalesce().indices()
    values = matrix.coalesce().values().unsqueeze(0)
    shape = matrix.coalesce().size()
    return {'indices':indices,'values':values, 'shape': shape}


def ensure_input_is_tensor(input):
    if input.is_sparse:
        input = sparse_to_tensor(input)
    return input


def torch_sparse_to_scipy_sparse(matrix):
    i = matrix.coalesce().indices().cpu()
    v = matrix.coalesce().values().cpu()

    (m, n) = matrix.shape[0], matrix.shape[1]
    return coo_matrix((v, i), shape=(m, n))


def scipy_sparse_to_torch_sparse(matrix):
    values = matrix.data
    indices = np.vstack((matrix.row, matrix.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse_coo_tensor(indices = i, values = v, size = matrix.shape )


def edge_to_node_matrix(edges, nodes, one_indexed=True):
    # very important to pay attention to if node indexes start with 1 or 0
    sigma1 = torch.zeros((len(nodes), len(edges)), dtype=torch.float)
    offset = int(one_indexed)
    j = 0
    for edge in edges:
        x, y = edge
        sigma1[x - offset][j] -= 1
        sigma1[y - offset][j] += 1
        j += 1
    return sigma1


def triangle_to_edge_matrix(triangles, edges):
    sigma2 = torch.zeros((len(edges), len(triangles)), dtype=torch.float)
    edges = [tuple(e) for e in edges]
    edges = {edges[i]: i for i in range(len(edges))}
    for l in range(len(triangles)):
        i, j, k = triangles[l]
        if (i, j) in edges:
            sigma2[edges[(i, j)]][l] += 1
        else:
            sigma2[edges[(j, i)]][l] -= 1

        if (j, k) in edges:
            sigma2[edges[(j, k)]][l] += 1
        else:
            sigma2[edges[(k, j)]][l] -= 1

        if (i, k) in edges:
            sigma2[edges[(i, k)]][l] -= 1
        else:
            sigma2[edges[(k, i)]][l] += 1

    return sigma2

def load_sim_data(data_path,shuffle_data):
    file = open(
        data_path,'rb')
    cochains = pickle.load(file)
    file.close()
    if shuffle_data:
        shuffle(cochains)
    return cochains


def return_L1_laplacians(cochain):
    coch = deepcopy(cochain)
    coch.convert_to_sparse()
    B1 = coch.B1
    B2 = coch.B2
    L1_upper = B2.mm(torch.transpose(B2, 0, 1))
    L1_lower = torch.transpose(B1, 0, 1).mm(B1)
    L1 = coch.L1
    return L1, L1_lower, L1_upper

def get_eigenvectors(cochain):
    L1, L1_lower, L1_upper = return_L1_laplacians(cochain)
    if torch.cuda.is_available():
        L1, L1_lower, L1_upper = L1.cuda(), L1_lower.cuda(), L1_upper.cuda()
    harm_space_eigenvalues, harm_space_eigenvectors, grad_space_eigenvalues, grad_space_eigenvectors, curl_space_eigenvalues, curl_space_eigenvectors = compute_hodge_eigv(L1, L1_lower, L1_upper)
    eigenv_dict = {"h_eigv": harm_space_eigenvectors,
                   "c_eigv": curl_space_eigenvectors,
                   "g_eigv": grad_space_eigenvectors}
    return eigenv_dict


def compute_hodge_eigv(L1, L1_lower, L1_upper):
    L1_lower_eig, L1_lower_eigvec = scipy.linalg.eig(L1_lower.to_dense().cpu().numpy())
    L1_upper_eig, L1_upper_eigvec = scipy.linalg.eig(L1_upper.to_dense().cpu().numpy())
    L1_eig, L1_eigvec = scipy.linalg.eig(L1.to_dense().cpu().numpy())

    L1_upper_eig = np.real_if_close(L1_upper_eig)
    L1_lower_eig = np.real_if_close(L1_lower_eig)
    L1_eig = np.real_if_close(L1_eig)

    small_upper = np.abs(L1_upper_eig) < 100 * np.finfo(L1_upper_eig.dtype).eps
    small_lower = np.abs(L1_lower_eig) < 100 * np.finfo(L1_lower_eig.dtype).eps
    small_l1 = np.abs(L1_eig) < 100 * np.finfo(L1_eig.dtype).eps
    L1_upper_eig[small_upper] = 0
    L1_lower_eig[small_lower] = 0
    L1_eig[small_l1] = 0

    harm_space_eigenvalues = torch.tensor(L1_eig[L1_eig == 0])
    harm_space_eigenvectors = torch.tensor(L1_eigvec[:, L1_eig == 0])
    grad_space_eigenvalues = torch.tensor(L1_lower_eig[L1_lower_eig != 0])
    grad_space_eigenvectors = torch.tensor(L1_lower_eigvec[:, L1_lower_eig != 0])
    curl_space_eigenvalues = torch.tensor(L1_upper_eig[L1_upper_eig != 0])
    curl_space_eigenvectors = torch.tensor(L1_upper_eigvec[:, L1_upper_eig != 0])

    if torch.cuda.is_available():
        harm_space_eigenvalues = harm_space_eigenvalues.cuda()
        harm_space_eigenvectors = harm_space_eigenvectors.cuda()
        grad_space_eigenvalues = grad_space_eigenvalues.cuda()
        grad_space_eigenvectors = grad_space_eigenvectors.cuda()
        curl_space_eigenvalues = curl_space_eigenvalues.cuda()
        curl_space_eigenvectors = curl_space_eigenvectors.cuda()
    return harm_space_eigenvalues, harm_space_eigenvectors, grad_space_eigenvalues, grad_space_eigenvectors, curl_space_eigenvalues, curl_space_eigenvectors
