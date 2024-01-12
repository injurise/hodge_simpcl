from __future__ import annotations
import src.loss_functions as L
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred import SETTINGS
import src.augmentors as A
from src.experiments.edge_flow_class.efc_utils import get_ef_paths_device, get_data,prepare_data_loader_ind_mat,compute_embedding_batch,compute_harm_emb_sim
from tqdm import tqdm
import src.models as models
from torch.optim import Adam
from src.contrast_models import DualBranchContrast
import numpy as np
from sacred.observers import FileStorageObserver
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import random
from sklearn.svm import LinearSVC
from src.utils import get_eigenvectors
from config import ROOT_DIR
import torch

def train(encoder_model, contrast_model, dataloader, optimizer, device):
    """
        Trains the encoder and contrast model.

        Args:
            encoder_model: The encoder object to be trained.
            contrast_model: The contrastive framework object used. E.g. Dual-Branch Framework
            dataloader: DataLoader providing the training data batches.
            optimizer: Optimizer used for updating the model weights.
            device: Device on which the computation is performed (e.g., 'cpu' or 'cuda').

        Returns:
            float: Total epoch loss after training on all batches.
        """
    encoder_model.train()
    epoch_loss = 0
    for batch in dataloader:
        batch.to(device)
        optimizer.zero_grad()

        # Forward pass through the encoder model. Returns embeddings of the origal (g) and of two augmented views (g1,g2).
        # Also returns the similarity score between the hodge embeddings in case a spectrally-weighted loss is used.
        g, g1, g2, sim_score = encoder_model(batch)

        # Passing the embeddings through the projection head specified in the model definition.
        g1, g2 = [encoder_model.encoder.project(embedding) for embedding in [g1, g2]]

        # Compute the contrastive loss and update model parameters.
        loss = contrast_model(g1=g1, g2=g2, batch=batch.features_batch, extra_neg_mask=sim_score)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() # Accumulate the loss over all batches.

    return epoch_loss

def test(encoder_model, dataloader, data_split, device='cpu'):
    """
        Evaluates the trained model on test data.
        Runs a linear SVM on the embeddings generated with the trained encoder model.

        Args:
            encoder_model: The encoder object containing the trained model.
            dataloader: DataLoader with all the data.
            data_split: Dictionary containing indices for train, validation, and test splits.
            device: Device on which the computation is performed (e.g., 'cpu' or 'cuda').

        Returns:
            Train, validation, and test accuracy.
        """
    encoder_model.eval()
    x = []
    y = []

    for batch in dataloader:
        batch.to(device)

        # Process each batch and obtain embeddings via a forward pass through the encoder model.
        num_nodes, num_edges, num_triangles = int(batch.num_n.sum()), int(batch.num_e.sum()), int(batch.num_t.sum())
        B1, B2 = torch.sparse_coo_tensor(batch.edge_index_s, batch.edge_vals_s,size=(num_nodes, num_edges)), \
                 torch.sparse_coo_tensor(batch.edge_index_t,batch.edge_vals_t, size=(num_edges, num_triangles))

        L1_lower = torch.transpose(B1, 0, 1).mm(B1)
        L1_upper = B2.mm(torch.transpose(B2, 0, 1))
        g = encoder_model.embed(batch.features, L1_lower.coalesce().indices(), L1_lower.coalesce().values(),
                                L1_upper.coalesce().indices(), L1_upper.coalesce().values(), batch.features_batch)
        x.append(g)
        y.append(batch.labels)
        batch.to("cpu")

    # Concatenates all the embeddings and labels.
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    # Setting up and training the SVM classifier.
    params = {'C': [0.000000001,0.000000005, 0.0000001,0.0000005, 0.00001,0.00005, 0.0001,0.0005, 0.001,0.005,
                    0.01,0.05, 0.1,0.5, 1,5, 10,50, 100, 500, 1000]}
    classifier = GridSearchCV(LinearSVC(max_iter=6000, dual=False), params, scoring='accuracy', verbose=1, cv=10)
    classifier.fit(x[data_split["train"]].detach().cpu().numpy(), y[data_split["train"]].detach().cpu().numpy())

    # Calculating accuracy for train, validation, and test splits.
    train_acc = accuracy_score(y[data_split["train"]].detach().cpu().numpy(),
                               classifier.predict(x[data_split["train"]].detach().cpu().numpy()))

    val_acc = accuracy_score(y[data_split["valid"]].detach().cpu().numpy(),
                                 classifier.predict(x[data_split["valid"]].detach().cpu().numpy()))

    test_acc = accuracy_score(y[data_split["test"]].detach().cpu().numpy(),
                                  classifier.predict(x[data_split["test"]].detach().cpu().numpy()))

    print(f"Train Accuracy:{train_acc}; Val Accuracy:{val_acc}; Test Accuracy:{test_acc}")
    return train_acc, val_acc, test_acc


class Encoder(torch.nn.Module):
    """
      Encoder class for processing edge flow data.

      Attributes:
          encoder: A model that encodes the input data.
          augmentor: Augmentations to be applied.
          eigenv_dict: Dictionary of eigenvalues and eigenvectors, used for spectral weighting.
      """

    def __init__(self, encoder, augmentor, eigenv_dict=False):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        self.eigenv_dict = eigenv_dict

    def forward(self, batch):
        """
        Forward pass for the Encoder. Two augmentations are applied to the original data point. For a single-branch
        approach one of the augmentations is the identity. In case of a spectrally-weighted loss, the similarity between the
        harmonic embeddings of the anchor and the augmented examples is computed.

        Args:
           batch: The input batch of data. (Batched Simplicial Complex)
        Returns:
           Embeddings of the original simplex and of the augemted simplexes, similarity scores between the harmonic embeddings
        """

        # Gets augmentors and extracts the incidence matrices for the batch
        aug1, aug2 = self.augmentor
        num_nodes, num_edges, num_triangles = int(batch.num_n.sum()), int(batch.num_e.sum()), int(batch.num_t.sum())
        B1, B2 = torch.sparse_coo_tensor(batch.edge_index_s, batch.edge_vals_s, size=(num_nodes, num_edges)), \
                 torch.sparse_coo_tensor(batch.edge_index_t, batch.edge_vals_t, size=(num_edges, num_triangles))

        # Apply augmentations to the batch. In this case of a dual branch contrastive model, we apply two augmentations.
        # For a single branch contrastive model, one of the augmentations is the identity.
        features_c1, B1_c1, B2_c1, num_n_c1, num_e_c1, num_t_c1, index = aug1(batch.features, B1, B2, batch.num_n,
                                                                              batch.num_e, batch.num_t,
                                                                              batch.batch_index)
        features_c2, B1_c2, B2_c2, num_n_c2, num_e_c2, num_t_c2, index = aug2(batch.features, B1, B2, batch.num_n,
                                                                              batch.num_e, batch.num_t,
                                                                              batch.batch_index)

        # Compute Hodge Laplacians for each branch and for the original simplicial complexes
        L1_lower_c1 = torch.transpose(B1_c1, 0, 1).mm(B1_c1)
        L1_upper_c1 = B2_c1.mm(torch.transpose(B2_c1, 0, 1))

        L1_lower_c2 = torch.transpose(B1_c2, 0, 1).mm(B1_c2)
        L1_upper_c2 = B2_c2.mm(torch.transpose(B2_c2, 0, 1))

        L1_lower = torch.transpose(B1, 0, 1).mm(B1)
        L1_upper = B2.mm(torch.transpose(B2, 0, 1))

        # The eigen_dict is non empty only when the loss is spectrally weighted. In this case, we compute the harmonic embeddings and their similarity scores.
        sim_score = None
        if self.eigenv_dict:
            # Compute the hodge embeddings and similarity scores for the harmonic embeddings
            harm_emb_anchor, curl_emb_anchor, grad_emb_anchor = compute_embedding_batch(features_c1, num_e_c1,
                                                                                        self.eigenv_dict["h_eigv"],
                                                                                        self.eigenv_dict["c_eigv"],
                                                                                        self.eigenv_dict["g_eigv"])

            harm_emb_sample, curl_emb_sample, grad_emb_sample = compute_embedding_batch(features_c2, num_e_c2,
                                                                                        self.eigenv_dict["h_eigv"],
                                                                                        self.eigenv_dict["c_eigv"],
                                                                                        self.eigenv_dict["g_eigv"])

            sim_score = 1 - compute_harm_emb_sim(emb_anchor=harm_emb_anchor, emb_sample=harm_emb_sample)
            epsilon = sim_score[sim_score > 0].min()  # torch.clamp(g, min=np.e ** (-1. / self.tau))
            sim_score = torch.clamp(sim_score, min=epsilon)
            # Multiply by N examples in the next line. This will later be divided away in the InfoNCE loss
            sim_score = sim_score / sim_score.sum(dim=1).view(sim_score.shape[0], 1) * sim_score.shape[1]

        # Encode the original data and the augmented data
        g = self.encoder(batch.features, L1_lower.coalesce().indices(), L1_lower.coalesce().values(),
                         L1_upper.coalesce().indices(), L1_upper.coalesce().values(), batch.features_batch)
        g1 = self.encoder(features_c1, L1_lower_c1.coalesce().indices(), L1_lower_c1.coalesce().values(),
                          L1_upper_c1.coalesce().indices(), L1_upper_c1.coalesce().values(), batch.features_batch)
        g2 = self.encoder(features_c2, L1_lower_c2.coalesce().indices(), L1_lower_c2.coalesce().values(),
                          L1_upper_c2.coalesce().indices(), L1_upper_c2.coalesce().values(), batch.features_batch)

        return g, g1, g2, sim_score

    def embed(self, x, low_ind, lower_v, up_ind, up_v, batch_features):
        return self.encoder(x, low_ind, lower_v, up_ind, up_v, batch_features)

# Use sacred to log experiments and save results.
SETTINGS['CAPTURE_MODE'] = 'sys'

ex = Experiment("Edge Flow Classification")
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(FileStorageObserver(f"{ROOT_DIR}/logs/trajectory_logs"))


#Configuration function for the experiment.
@ex.config
def run_cfg():
    dataset = 'trajectories'
    model_type = "CSCN"
    model_params = {'hidden_dim': 64, 'hidden_layers': 3, 'n_output_features': 64, 'projection_head_dim': 126}
    epochs = 200
    batch_size = 100  # 32
    lr = 0.001
    weight_decay = 0.0001
    loss_type = "InfoNCE"
    loss_params = {"tau": 0.2}
    augmentations = {"SpectralEdgeDrop": {"max_drop_ratio": 0.4, "lr_spec": 0.02, "iteration": 75, "initialisation": "constant",
                             "max_lr_change": 'automatic'}}
    run_n = 1  # this is just to keep track of repeated runs
    spectral_weighting = True

@ex.automain
def main(dataset, model_type, model_params, augmentations, epochs, loss_type, loss_params, lr, weight_decay,
         batch_size, spectral_weighting, run_n):

    # Retrieving data paths and setting computation device
    train_datapath, val_datapath, device = get_ef_paths_device(dataset)

    # Setting seeds for reproducibility
    random.seed(run_n)
    torch.manual_seed(run_n)
    np.random.seed(run_n)

    # Data loading and preparation. For evaluation we use half of the validation set as a test set. This is specified in the data_split dictionary.
    cochains, data_split, shuffle_indices = get_data(train_datapath, val_datapath, shuffle_data=True)
    train_loader = prepare_data_loader_ind_mat(cochains, batch_size=batch_size, shuffle=False)
    test_loader = prepare_data_loader_ind_mat(cochains, batch_size=batch_size, shuffle=False)

    # Spectral weighting setup
    eigenv_dict = {}
    if spectral_weighting:
        eigenv_dict = get_eigenvectors(cochain = cochains[data_split["train"][0]])   # Take any simp_complex -> Laplacians are the same for these datasets

    # Augmentation setup
    aug_1_params,aug_2_params = augmentations.copy(),augmentations.copy()
    if "SpectralEdgeDrop" in augmentations:
        # delete spectral edge drop from aug_1_params
        aug_1_params['Identity'] = {}
        del aug_1_params["SpectralEdgeDrop"]

        # load probabilities for spectral edge drop 2
        spec_edge_drop_aug_2 = getattr(A, "SpectralEdgeDrop")(**aug_2_params["SpectralEdgeDrop"])
        drop_prob_file = f"drop_prob_lr_{spec_edge_drop_aug_2.lr_spec}_iter_{spec_edge_drop_aug_2.iteration}_max_drop_ratio_{spec_edge_drop_aug_2.max_drop_ratio}_init_{spec_edge_drop_aug_2.initialisation}_max_lr_change_{spec_edge_drop_aug_2.max_lr_change}.pt"
        drop_prob_path = f"{ROOT_DIR}/data/{dataset}/precal_probabilities/{drop_prob_file}"
        spec_edge_drop_aug_2.load_prob(path=drop_prob_path)
        shuf_drop_prob = torch.tensor(np.array(spec_edge_drop_aug_2.drop_prob)[shuffle_indices])
        spec_edge_drop_aug_2.drop_prob = list(shuf_drop_prob)
        del aug_2_params["SpectralEdgeDrop"]

    augs1 = [getattr(A, aug)(**params) if params else getattr(A, aug)() for aug, params in aug_1_params.items()]
    augs2 = [getattr(A, aug)(**params) if params else getattr(A, aug)() for aug, params in aug_2_params.items()]
    if "SpectralEdgeDrop" in augmentations:
        augs2.append(spec_edge_drop_aug_2)

    aug1 = A.Compose(augs1)
    aug2 = A.Compose(augs2)

    # Model & Loss definition
    model_class = getattr(models, model_type)
    num_features = train_loader.dataset[0].features.shape[1]
    model = model_class(**model_params, n_edge_features=num_features)
    loss_class = getattr(L, loss_type)
    loss = loss_class(**loss_params)

    # Encoder and Contrastive Framework
    encoder_model = Encoder(encoder=model, augmentor=(aug1, aug2), eigenv_dict=eigenv_dict).to(device)
    contrast_model = DualBranchContrast(loss=loss, mode='G2G').to(device)
    optimizer = Adam(encoder_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5, patience=10,min_lr=0.0000001)

    # Training and evaluation loop
    best_loss,cnt_wait = 1e9,0
    with tqdm(total=epochs, desc='(T)') as pbar:
        for epoch in range(1, epochs + 1):
            lr = scheduler.optimizer.param_groups[0]['lr']
            loss = train(encoder_model, contrast_model, train_loader, optimizer, device=device)
            ex.log_scalar("Loss", loss)
            pbar.set_postfix({'loss': loss})
            pbar.update()

            # Model evaluation
            train_result, val_result, test_result = test(encoder_model, test_loader, data_split,device=device)
            ex.log_scalar("Train Accuracy", train_result)
            ex.log_scalar("Test Accuracy", test_result)
            if val_result is not None:
                ex.log_scalar("Val Accuracy", val_result)

            scheduler.step(loss)

            # Check for early stopping
            if (loss < best_loss) and (epoch % 1 == 0):
                best_loss = loss
                cnt_wait = 0
            else:
                cnt_wait += 1
            if cnt_wait == 50:
                print('Early stopping!')
                break
