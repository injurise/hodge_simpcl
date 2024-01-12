import torch
from torch_geometric.data import Data
import numpy as np
from src.utils import load_sim_data
import torch.nn.functional as F
from config import ROOT_DIR
from torch_geometric.loader import DataLoader


def get_ef_paths_device(dataset):
    """
       Gets the paths for the training and validation data and determines the computation device.

       Args:
           dataset (str): The name of the dataset.

       Returns:
        Paths for training data, validation data, and the computation device.
       """

    train_datapath = f"{ROOT_DIR}/data/{dataset}/{dataset}_cochains_train.obj"
    val_datapath = f"{ROOT_DIR}/data/{dataset}/{dataset}_cochains_val.obj"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return train_datapath, val_datapath, device


class PairData(Data):
    """
       Custom Data class in PyTorch Geometric which is used to store the two boundary operators for the simplicial data.
       Link:  https: // pytorch - geometric.readthedocs.io / en / latest / advanced / batching.html

       Attributes:
           edge_index_s (Tensor): Inidces of "Lower" boundary operator.
           edge_vals_s (Tensor): Values of "Lower" boundary operator.
           features (Tensor): Feature values of flows observed.
           edge_index_t (Tensor): Indices of "Upper" boundary operator.
           edge_vals_t (Tensor): Indices of "Lower" boundary operator.
           num_n (int): Number of nodes.
           num_e (int): Number of edges.
           num_t (int): Number of triangles or higher order structures.
           labels (Tensor): Labels for the data.
           batch_index (Tensor): Batch index for the data.
       """
    def __init__(self, edge_index_s=None, edge_vals_s=None, features=None, edge_index_t=None, edge_vals_t=None,
                 num_n=None, num_e=None, num_t=None, labels=None, batch_index=None):
        super().__init__()

        self.edge_index_s = edge_index_s
        self.edge_vals_s = edge_vals_s
        self.features = features
        self.num_n = num_n
        self.num_e = num_e
        self.num_t = num_t
        self.labels = labels
        self.edge_index_t = edge_index_t
        self.edge_vals_t = edge_vals_t
        self.batch_index = batch_index
        # self.x_t = x_t

    def __inc__(self, key, value, *args, **kwargs):
        # Method to define how to increment indices when batching multiple boundary operators.
        if key == 'edge_index_s':
            return torch.tensor([[self.num_n], [self.num_e]])
        if key == 'edge_index_t':
            return torch.tensor([[self.num_e], [self.num_t]])
        if key == 'batch_index':
            return torch.tensor([0])
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def to(self, device):
        # Moves all tensors of the data to the specified device (e.g., GPU or CPU).
        self.edge_index_s = self.edge_index_s.to(device)
        self.edge_vals_s = self.edge_vals_s.to(device)
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.edge_index_t = self.edge_index_t.to(device)
        self.edge_vals_t = self.edge_vals_t.to(device)
        if hasattr(self, 'features_batch'):
            self.features_batch = self.features_batch.to(device)
            self.batch_index = self.batch_index.to(device)


def get_data(train_datapath, val_datapath, shuffle_data=True):
    """
    Loads the training and validation data and combines them into one dataset for the transductive unsupervised setting.
    The labels are split into training,validation and test labels. Hereby half of the "validation" data is used for testing.

    Args:
        train_datapath (str): Path to the training data.
        val_datapath (str): Path to the validation data.
        shuffle_data (bool): Whether to shuffle the data.

    Returns:
       The combined training and validation data, the data split indices with labels for the supervised learner, and shuffle indices.
    """
    # Load data from training and validation paths
    train_cochains = load_sim_data(train_datapath, shuffle_data=False)
    val_cochains = load_sim_data(val_datapath, shuffle_data=False)

    # Determine the number of samples in training and validation sets
    n_train = len(train_cochains)
    n_val = len(val_cochains)

    # Create indices for splitting the labels for the supervised evaluation on the embeddings
    indices = torch.arange(n_val + n_train)
    split = {
        'train': indices[:n_train],
        'valid': indices[n_train:(n_train + int(n_val / 2))],
        "test": indices[(n_train + int(n_val / 2)):]}

    # Combine data for the transductive contrastive setting
    train_cochains.extend(val_cochains)

    # Shuffle if specified
    shuffle_indices = np.arange(len(train_cochains))
    if shuffle_data:
        np.random.shuffle(shuffle_indices)
        train_cochains = np.array(train_cochains)
        train_cochains = list(train_cochains[shuffle_indices])

    return train_cochains, split, shuffle_indices


def compute_embedding_batch(features, edge_list, harm_space_eigenvectors, curl_space_eigenvectors,
                            grad_space_eigenvectors):
    """
    Computes harmonic, curl, and gradient embeddings for a batch of edge flows.

    Args:
        features (torch.Tensor): The features of the edge flows.
        edge_list (List[int]): List of numbers of features for each edge flow in the batch.
        harm_space_eigenvectors (torch.Tensor): Eigenvectors for harmonic space.
        curl_space_eigenvectors (torch.Tensor): Eigenvectors for curl space.
        grad_space_eigenvectors (torch.Tensor): Eigenvectors for gradient space.

    Returns:
        Harmonic, curl, and gradient embeddings.
    """
    harm_emb_batch = []
    curl_emb_batch = []
    grad_emb_batch = []
    start = 0
    for n_edge in edge_list:
        end = start + n_edge
        edge_flow = features[start:end]

        # Computing embeddings
        harmonic_embedding = torch.real(torch.mm(torch.t(harm_space_eigenvectors.type(torch.complex64)), edge_flow.type(torch.complex64)))
        curl_embedding = torch.real(torch.mm(torch.t(curl_space_eigenvectors), edge_flow.type(torch.complex64)))
        gradient_embedding = torch.real(torch.mm(torch.t(grad_space_eigenvectors), edge_flow.type(torch.complex64)))

        harm_emb_batch.append(harmonic_embedding)
        curl_emb_batch.append(curl_embedding)
        grad_emb_batch.append(gradient_embedding)
        start = end

    # Concatenating the embeddings for the batch
    harm_emb_batch = torch.t(torch.cat(harm_emb_batch, dim=1))
    curl_emb_batch = torch.t(torch.cat(curl_emb_batch, dim=1))
    grad_emb_batch = torch.t(torch.cat(grad_emb_batch, dim=1))

    return harm_emb_batch, curl_emb_batch, grad_emb_batch


def compute_harm_emb_sim(emb_anchor, emb_sample, normalise=False):
    """
        Computes the cosine similarity between two sets of embeddings.

        Args:
            emb_anchor (torch.Tensor): The embedding of the anchor.
            emb_sample (torch.Tensor): The embedding to compare against the anchor.
            normalise (bool, optional): Whether to normalize the similarity scores. Defaults to False.

        Returns:
            A tensor representing the similarity scores between the anchor and sample embeddings.
        """
    emb_anchor, emb_sample = F.normalize(emb_anchor), F.normalize(emb_sample)
    harm_sim = emb_anchor.mm(emb_sample.t())
    if normalise:
        harm_sim = (2 * (harm_sim - torch.min(harm_sim))) / (torch.max(harm_sim) - torch.min(harm_sim)) - 1

    return harm_sim


def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    """
       Computes the cosine similarity between two tensors.

       Args:
           h1 (torch.Tensor): The first tensor.
           h2 (torch.Tensor): The second tensor.

       Returns:
           torch.Tensor: A tensor representing the cosine similarity.
       """
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    sim = h1 @ h2.t()
    return sim


def prepare_data_loader_ind_mat(cochains, batch_size=32, shuffle=True):
    """
       Prepares a DataLoader for a given list of cochains.

       Args:
           cochains (List): A list of cochains to be processed.
           batch_size (int, optional): The size of each batch. Defaults to 32.
           shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

       Returns:
           DataLoader: A DataLoader object ready for iteration.
       """
    data_list = []
    for idx, cochain in enumerate(cochains):
        cochain.convert_to_sparse()
        # Creating a PairData object for each cochain. This is necessary as we iterate over indeces of both B1,B2
        data = PairData(cochain.B1.coalesce().indices(), cochain.B1.coalesce().values(), cochain.X1,
                        cochain.B2.coalesce().indices(), cochain.B2.coalesce().values(), int(cochain.B1.shape[0]),
                        int(cochain.B1.shape[1]), int(cochain.B2.shape[1]),
                        labels=cochain.labels, batch_index=torch.tensor([idx]))
        data_list.append(data)

    loader = DataLoader(data_list, batch_size=batch_size, follow_batch=['features'], shuffle=shuffle)
    return loader
