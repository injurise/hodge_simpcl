import numpy as np
import torch
import os
import argparse
from src.utils import load_sim_data
from src.augmentors.spec_edge_drop import SpectralEdgeDrop
from src.utils import (return_L1_laplacians,compute_hodge_eigv)
from copy import deepcopy
from config import ROOT_DIR

def calculate_edge_drop_probs(train_datapath, val_datapath, drop_prob_dir, lr_spec, iteration, max_drop_ratio,
                              initialisation, max_lr_change):

    cochains = load_sim_data(train_datapath, shuffle_data=False)
    cochains.extend(load_sim_data(val_datapath, shuffle_data=False))
    coch = deepcopy(cochains[0])
    coch.convert_to_sparse()
    L1, L1_lower, L1_upper = return_L1_laplacians(coch)

    harm_space_eigenvalues, harm_space_eigenvectors, grad_space_eigenvalues, \
    grad_space_eigenvectors, curl_space_eigenvalues, curl_space_eigenvectors = compute_hodge_eigv(L1, L1_lower, L1_upper)

    spec_aug_edge_drop = SpectralEdgeDrop(max_drop_ratio=max_drop_ratio,
                                          lr_spec=lr_spec,
                                          iteration=iteration,
                                          threshold=1,
                                          max_lr_change=max_lr_change,
                                          initialisation=initialisation)

    cochain_losses = []
    for cochain in cochains:
        cochain.convert_to_sparse()
        co_loss = spec_aug_edge_drop.calc_hodge_prob(
            x=cochain.X1.cpu(), harm_space_eigenvectors=harm_space_eigenvectors.cpu(),
            curl_space_eigenvectors=curl_space_eigenvectors.cpu(), grad_space_eigenvectors=grad_space_eigenvectors.cpu()
        )
        cochain_losses.append(co_loss)

    result = {
        'Learning Rate': lr_spec,
        'Iterations': iteration,
        'Ratio': max_drop_ratio,
        "Initialisation": initialisation,
        "Max LR Change": max_lr_change,
        'Mean Loss': np.mean(cochain_losses),
        'Min Loss': np.min(cochain_losses),
        'Max Loss': np.max(cochain_losses)
    }
    print(result)

    # Saving the drop_prob_list to a separate file for each combination of parameters
    drop_prob_file_name = f"drop_prob_lr_{lr_spec}_iter_{iteration}_max_drop_ratio_{max_drop_ratio}_init_{initialisation}_max_lr_change_{max_lr_change}.pt"
    drop_prob_file_path = os.path.join(drop_prob_dir, drop_prob_file_name)
    torch.save(spec_aug_edge_drop.drop_prob, drop_prob_file_path)

if __name__ == '__main__':

    drop_prob_dir = f"{ROOT_DIR}/data/trajectories/precal_probabilities"
    train_datapath = f"{ROOT_DIR}/data/trajectories/trajectories_cochains_train.obj"
    val_datapath = f"{ROOT_DIR}/data/trajectories/trajectories_cochains_val.obj"

    config_dict = {
        "train_datapath": train_datapath,
        "val_datapath": val_datapath,
        "drop_prob_dir": drop_prob_dir,
        "lr_spec": 0.02,
        "iteration": 75,
        "max_drop_ratio": 0.4,
        "initialisation": "constant",
        "max_lr_change": "automatic"}

    calculate_edge_drop_probs(**config_dict)
