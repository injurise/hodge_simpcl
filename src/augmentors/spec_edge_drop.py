import numpy as np
import torch
from torch.nn.parameter import Parameter
from tqdm import tqdm
from src.augmentors.augmentor import SAugmentor
from torch.distributions.bernoulli import Bernoulli


def bisection(a, b, n_perturbations, epsilon, adj_changes, threshold):
    def func(x):
        return torch.clamp(adj_changes - x, 0, threshold).sum() - n_perturbations
    # Bisection method
    miu = a
    while ((b - a) >= epsilon):
        miu = (a + b) / 2
        # Check if middle point is root
        if (func(miu) == 0.0):
            b = miu
            break
        # Decide the side to repeat the steps
        if (func(miu) * func(a) < 0):
            b = miu
        else:
            a = miu
    # print("The value of root is : ","%.4f" % miu)
    return miu


def projection(n_perturbations, adj_changes, threshold=0.5):
    """
    Projects the changes in the adjacency matrix to meet the perturbation budget.

    Args:
        n_perturbations (int): The number of perturbations to apply.
        adj_changes (Tensor): Tensor representing changes in adjacency.
        threshold (float, optional): The threshold for clamping. Defaults to 0.5.

    Returns:
        Tensor: The projected adjacency changes.
    """

    if torch.clamp(adj_changes, 0, threshold).sum() > n_perturbations:
        left = (adj_changes).min()
        right = adj_changes.max()
        miu = bisection(left, right, n_perturbations, 1e-4, adj_changes, threshold)
        l = left.cpu().detach()
        r = right.cpu().detach()
        m = miu.cpu().detach()
        adj_changes.data.copy_(torch.clamp(adj_changes.data - miu, min=0, max=1))
    else:
        adj_changes.data.copy_(torch.clamp(adj_changes.data, min=0, max=1))
    return adj_changes


class SpectralEdgeDrop(SAugmentor):
    """
        An augmentation class for dropping edges in a graph based on spectrally optimized probabilities.

        Args:
            max_drop_ratio (float): The maximum ratio of edges to drop (budget).
            lr_spec (float): Learning rate for spectral adjustments.
            iteration (int): Number of iterations for the optimization.
            device (str, optional): The device to run the computations on. Defaults to "cpu".
            threshold (float, optional): Threshold for max. edge dropping probability. Defaults to 1.
            max_lr_change (float, optional): If specified, adjusts the learning rate for optimization dynamically to the specified max. change in drop probability.
                                             Defaults to 0.15. Can be chosen to be adjusted automatically, by specifiying "automatic".
            initialisation (str, optional): Method for initializing probabilities. Defaults to "uniform".
        """
    def __init__(self, max_drop_ratio, lr_spec, iteration, device="cpu",threshold=1, max_lr_change=0.15, initialisation="uniform"):
        super(SpectralEdgeDrop, self).__init__()

        self.max_drop_ratio = max_drop_ratio
        self.lr_spec = lr_spec
        self.iteration = iteration
        self.device = device
        self.threshold = threshold
        self.drop_prob = []  # probabilities to drop an edge. If 1 then the edge is drop 100 % of the time
        self.max_lr_change = max_lr_change
        self.initialisation = initialisation

    def load_prob(self, path):
        self.drop_prob = torch.load(path)

    def calc_hodge_prob(self, x, curl_space_eigenvectors, grad_space_eigenvectors, harm_space_eigenvectors,
                        verbose=False, silence=False, device="cpu"):
        """
                Optimizes the dropout probabilities based on Hodge decomposition.

                Args:
                    x (Tensor): The edge features.
                    curl_space_eigenvectors (Tensor): Eigenvectors for curl space.
                    grad_space_eigenvectors (Tensor): Eigenvectors for gradient space.
                    harm_space_eigenvectors (Tensor): Eigenvectors for harmonic space.
                    verbose (bool, optional): Enables verbose output. Defaults to False.
                    silence (bool, optional): Disables tqdm progress bar if True. Defaults to False.
                    device (str, optional): The device to run the computations on. Defaults to "cpu".

                Returns:
                    float: The  loss after the final iteration.
        """
        if self.max_lr_change == "automatic":
            self.max_lr_change = self.max_drop_ratio / 4

        harm_embedding = torch.real(
            torch.mm(torch.t(harm_space_eigenvectors.type(torch.complex64)), x.type(torch.complex64)))
        curl_embedding = torch.real(
            torch.mm(torch.t(curl_space_eigenvectors.type(torch.complex64)), x.type(torch.complex64)))
        grad_embedding = torch.real(
            torch.mm(torch.t(grad_space_eigenvectors.type(torch.complex64)), x.type(torch.complex64)))

        x = x.to_sparse()
        probs = Parameter(torch.FloatTensor(x.values().shape), requires_grad=True).to(device)

        if self.initialisation == "uniform":
            torch.nn.init.uniform_(probs)
        if self.initialisation == "constant":
            torch.nn.init.constant_(probs, self.max_drop_ratio)

        dropout_change_vec = torch.sparse_coo_tensor(x.indices(), probs, size=x.size())
        n_pertubation_budget = int((self.max_drop_ratio) * len(probs.detach()))
        if n_pertubation_budget == 0:
            n_pertubation_budget = 1

        if self.max_lr_change != None:
            max_lr_change = self.max_lr_change

        with tqdm(total=self.iteration, disable=silence) as pbar:
            verb = max(1, int(self.iteration / 10))
            for t in range(1, self.iteration + 1):
                ones_sparse = torch.sparse_coo_tensor(dropout_change_vec.coalesce().indices(),
                                                      torch.ones(dropout_change_vec.coalesce().values().shape[0]),
                                                      dropout_change_vec.shape)
                expected_flow = x * (ones_sparse - dropout_change_vec)
                exp_grad_embedding = torch.mm(torch.t(torch.real(grad_space_eigenvectors)), expected_flow.to_dense())
                exp_harm_embedding = torch.mm(torch.t(torch.real(harm_space_eigenvectors)), expected_flow.to_dense())
                exp_curl_embedding = torch.mm(torch.t(torch.real(curl_space_eigenvectors)), expected_flow.to_dense())

                exp_mat = torch.mm(expected_flow, torch.t(expected_flow)).to_dense()
                diag = torch.diag(exp_mat)
                idx = (torch.t(dropout_change_vec.to_dense()) > 0)
                diag[idx.squeeze(0)] = diag[idx.squeeze(0)] / torch.t(dropout_change_vec.to_dense())[idx]
                exp_mat[range(len(exp_mat)), range(len(exp_mat))] = diag

                ## compute grad component matrix (this is the matrix left if the loss expression is expanded)
                grad_exp_mat = torch.mm(torch.t(torch.real(grad_space_eigenvectors)), exp_mat)
                grad_exp_mat = torch.mm(grad_exp_mat, torch.real(grad_space_eigenvectors))
                grad_exp_mat = torch.trace(grad_exp_mat).to_sparse()

                ## compute curl component matrix
                curl_exp_mat = torch.mm(torch.t(torch.real(curl_space_eigenvectors)), exp_mat)
                curl_exp_mat = torch.mm(curl_exp_mat, torch.real(curl_space_eigenvectors))
                curl_exp_mat = torch.trace(curl_exp_mat).to_sparse()
                ## compute harm component matrix
                harm_exp_mat = torch.mm(torch.t(torch.real(harm_space_eigenvectors)), exp_mat)
                harm_exp_mat = torch.mm(harm_exp_mat, torch.real(harm_space_eigenvectors))
                harm_exp_mat = torch.trace(harm_exp_mat).to_sparse()

                ## compute the different parts of the final loss expression
                mse_grad = torch.linalg.vector_norm(grad_embedding).pow(2) - torch.mm(torch.t(grad_embedding),
                           exp_grad_embedding).squeeze() - torch.mm(torch.t(exp_grad_embedding), grad_embedding).squeeze() \
                           + grad_exp_mat.values().squeeze()
                mse_grad = mse_grad / torch.linalg.vector_norm(grad_embedding)

                mse_curl = torch.linalg.vector_norm(curl_embedding).pow(2) - torch.mm(torch.t(curl_embedding),
                            exp_curl_embedding).squeeze() - torch.mm(torch.t(exp_curl_embedding), curl_embedding).squeeze() \
                           + curl_exp_mat.values().squeeze()
                mse_curl = mse_curl / torch.linalg.vector_norm(curl_embedding)

                mse_harm = torch.linalg.vector_norm(harm_embedding).pow(2) - torch.mm(torch.t(harm_embedding),exp_harm_embedding).squeeze() \
                           - torch.mm(torch.t(exp_harm_embedding), harm_embedding).squeeze() + harm_exp_mat.values().squeeze()
                mse_harm = mse_harm / torch.linalg.vector_norm(harm_embedding)

                # calculate the final loss
                reg_loss = mse_grad - mse_harm - mse_curl
                print(f"grad_mse:{mse_grad}, harm_mse:{mse_harm}, curl_mse:{mse_curl} ")

                # Optimize the probabilities with respect to the final loss by taking gradients
                loss = reg_loss
                torch.autograd.set_detect_anomaly(True)
                adj_grad = torch.autograd.grad(loss, dropout_change_vec)[0]

                if t % 5 == 0:
                    if self.max_lr_change != None:
                        max_lr_change = max_lr_change / 2
                    else:
                        lr = self.lr_spec / np.sqrt(t / 5 + 1)

                max_adj_grad = torch.max(torch.abs(adj_grad).to_dense())
                if self.max_lr_change != None:
                    lr = max_lr_change / max_adj_grad
                dropout_change_vec = dropout_change_vec + lr * adj_grad
                dropout_change_vec = dropout_change_vec.to_dense()

                # Project the probabilities to the valid range and such that they meet the budget
                before_p = torch.clamp(dropout_change_vec, 0, 1).sum()
                before_l = dropout_change_vec.min()
                before_r = dropout_change_vec.max()
                before_m = torch.clamp(dropout_change_vec, 0, 1).sum() / torch.count_nonzero(dropout_change_vec)
                dropout_change_vec = projection(n_pertubation_budget, dropout_change_vec, threshold=self.threshold)
                after_p = dropout_change_vec.sum()
                after_l = dropout_change_vec.min()
                after_r = dropout_change_vec.max()
                after_m = dropout_change_vec.sum() / torch.count_nonzero(dropout_change_vec)
                dropout_change_vec = dropout_change_vec.to_sparse()

                if verbose and t % verb == 0:
                    print(
                        '-- Epoch {}, '.format(t),
                        'reg loss = {:.4f} | '.format(reg_loss),
                        'ptb budget/b/a = {:.1f}/{:.1f}/{:.1f}'.format(n_pertubation_budget, before_p, after_p),
                        'min b/a = {:.4f}/{:.4f}'.format(before_l, after_l),
                        'max b/a = {:.4f}/{:.4f}'.format(before_r, after_r),
                        'mean b/a = {:.4f}/{:.4f}'.format(before_m, after_m))

                pbar.set_postfix(
                    {'reg_loss': reg_loss.item(), 'budget': n_pertubation_budget, 'b_proj': before_p.item(),
                     'a_proj': after_p.item()})
                pbar.update()

            self.drop_prob.append(dropout_change_vec.to_dense().detach().cpu())
            return reg_loss.item()

    def augment(self, x, B1, B2, num_n, num_e, num_t, index):
        """
               Applies the augmentation by dropping edges based on the computed probabilities.

               Args:
                   x (Tensor): The edge features.
                   B1 (Tensor): The Lower incidence matrix.
                   B2 (Tensor): The Upperincidence matrix.
                   num_n, num_e, num_t (int): Numbers of nodes, edges, and triangles, respectively.
                   index (Tensor): The indices of the data points.

               Returns:
                   tuple: The augmented edge features, B1, B2, and the numbers of nodes, edges, and triangles.
               """
        if len(self.drop_prob) == 1:
            probs = self.drop_prob[index]
        else:
            probs = []
            for i in index:
                probs.append(self.drop_prob[i])
            probs = torch.cat(probs)

        dist = Bernoulli(probs)
        samples = dist.sample()
        if torch.cuda.is_available():
            samples = samples.to('cuda')

        x = x * (1 - samples)

        return x, B1, B2, num_n, num_e, num_t, index
