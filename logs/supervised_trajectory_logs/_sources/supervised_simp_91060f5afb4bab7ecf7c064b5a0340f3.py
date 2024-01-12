from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred import SETTINGS
import src.models as models
from tqdm import tqdm
import random
from src.experiments.edge_flow_class.efc_utils import get_ef_paths_device, get_data, prepare_data_loader_ind_mat
import torch
import numpy as np
from sacred.observers import FileStorageObserver
from config import ROOT_DIR

def test(model, dataset):
    """
        Evaluate the model's accuracy on a given dataset.

        Args:
            model (torch.nn.Module): The neural network model to be evaluated.
            dataset (DataLoader): The dataset to evaluate the model on, wrapped in a DataLoader.

        Returns:
            The accuracy of the model on the provided dataset.
    """
    model.eval()
    correct = 0
    datapoints = 0
    for batch in dataset:
        # Move the batch to CUDA if available
        if torch.cuda.is_available():
            batch.to("cuda")

        with torch.no_grad(): # Disable gradient calculations
            # Extract component from the batch
            num_nodes, num_edges, num_triangles = int(batch.num_n.sum()), int(batch.num_e.sum()), int(batch.num_t.sum())
            B1, B2 = torch.sparse_coo_tensor(batch.edge_index_s, batch.edge_vals_s,size=(num_nodes, num_edges)), torch.sparse_coo_tensor(batch.edge_index_t, batch.edge_vals_t, size=(num_edges, num_triangles))

            # Compute Hodge Laplacians
            L1_lower = torch.transpose(B1, 0, 1).mm(B1)
            L1_upper = B2.mm(torch.transpose(B2, 0, 1))

            # Make predictions
            pred = model(x=batch.features, lower_edge_index=L1_lower.coalesce().indices(),
                         lower_edge_values=L1_lower.coalesce().values(),
                         upper_edge_index=L1_upper.coalesce().indices(),
                         upper_edge_values=L1_upper.coalesce().values(),
                         batch_index=batch.features_batch).cpu()

            # Update correct predictions count
            correct += float((torch.argmax(pred, 1).flatten() == batch.labels.detach().cpu()).type(torch.float).sum().item())
            datapoints += float(pred.shape[0])

    acc = correct / datapoints

    return acc

# Configure Sacred for Logging
SETTINGS['CAPTURE_MODE'] = 'sys'

ex = Experiment("SupGCN")
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(FileStorageObserver(f"{ROOT_DIR}/logs/supervised_trajectory_logs"))

@ex.config
def run_cfg():
    model_type = 'SupSCN'  # 'SupSCN'
    model_params = {'n_edge_features': 1,'hidden_dim': 64, 'hidden_layers': 3,'n_output_features': 2}
    epochs = 150
    batch_size = 64
    lr = 0.0005
    weight_decay = 0.0001
    dataset = "trajectories"
    run_n = 2  # this is just to keep track of repeated runs

@ex.automain
def main(model_type, model_params, epochs, batch_size, lr, weight_decay, dataset, run_n):

    # Set seeds for reproducibility
    random.seed(run_n)
    torch.manual_seed(run_n)
    np.random.seed(run_n)

    #Load data
    train_datapath, val_datapath, device = get_ef_paths_device(dataset)
    cochains, data_split, shuffle_indices = get_data(train_datapath, val_datapath, shuffle_data=True)

    # Prepare DataLoader for train, validation, and test sets
    train_cochains = [cochains[i] for i in data_split['train']]
    val_cochains = [cochains[i] for i in data_split['valid']]
    test_cochains = [cochains[i] for i in data_split['test']]
    train_loader = prepare_data_loader_ind_mat(train_cochains, batch_size=batch_size, shuffle=False)
    val_loader = prepare_data_loader_ind_mat(val_cochains, batch_size=batch_size, shuffle=False)
    test_loader = prepare_data_loader_ind_mat(test_cochains, batch_size=batch_size, shuffle=False)

    # Initialize the model and optimizer
    model_class = getattr(models, model_type)
    model = model_class(**model_params)
    if torch.cuda.is_available():
        model = model.cuda()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=10,min_lr=0.0000001)
    loss_f = torch.nn.CrossEntropyLoss()

    # Training Loop
    best_val_acc = 0
    with tqdm(total=epochs, desc='(T)') as pbar:
        for j in range(epochs):
            epoch_loss = 0
            lr = scheduler.optimizer.param_groups[0]['lr']
            model.train()
            for batch in train_loader:
                if torch.cuda.is_available():
                    batch.to("cuda")

                # Extract Components from batch
                num_nodes, num_edges, num_triangles = int(batch.num_n.sum()), int(batch.num_e.sum()), int(batch.num_t.sum())
                B1, B2 = torch.sparse_coo_tensor(batch.edge_index_s, batch.edge_vals_s,size=(num_nodes, num_edges)),\
                         torch.sparse_coo_tensor(batch.edge_index_t, batch.edge_vals_t, size=(num_edges, num_triangles))

                #Compute Hodge Laplacians
                L1_lower = torch.transpose(B1, 0, 1).mm(B1)
                L1_upper = B2.mm(torch.transpose(B2, 0, 1))

                optimiser.zero_grad()

                #Forward pass
                prediction = model(x=batch.features, lower_edge_index=L1_lower.coalesce().indices(),
                                   lower_edge_values=L1_lower.coalesce().values(),
                                   upper_edge_index=L1_upper.coalesce().indices(),
                                   upper_edge_values=L1_upper.coalesce().values(),
                                   batch_index=batch.features_batch)

                # Calculate Loss and Compute Gradients
                loss = loss_f(prediction, batch.labels)
                epoch_loss += float(loss.item())
                loss.backward()
                optimiser.step()

            ex.log_scalar("Loss", epoch_loss)
            scheduler.step(epoch_loss)
            pbar.set_postfix({'loss': epoch_loss})
            pbar.update()

            # Evaluate the model on training, validation, and test sets
            with torch.no_grad():
                train_acc = test(model, train_loader) * 100
                val_acc = test(model, val_loader) * 100
                test_acc = test(model, test_loader) * 100

                ex.log_scalar("Train Accuracy", train_acc)
                ex.log_scalar("Val Accuracy", val_acc)
                ex.log_scalar("Test Accuracy", test_acc)
                print(f"Epoch Loss: {epoch_loss:.4f}; "
                      f"Train Acc: {train_acc:.4f};"
                      f" Val Acc: {val_acc:.4f}; "
                      f"Test Acc: {test_acc:.4f} "
                      f"for epoch {j}")

