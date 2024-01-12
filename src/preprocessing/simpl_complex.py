import torch
from src.utils import sparse_to_tensor, ensure_input_is_tensor, tensor_to_sparse

class simplicial_data:
    """
      Represents basic data for a simplicial complex.

      Args:
          X0 (Tensor): Node features.
          X1 (Tensor): Edge features.
          X2 (Tensor): Triangle features.
          labels (Tensor): Labels for the data.
      """
    def __init__(self, X0, X1, X2, labels):
        self.X0 = X0.to(torch.float)
        self.X1 = X1.to(torch.float)
        self.X2 = X2.to(torch.float)

        self.n_nodes = X0.shape[0]
        self.n_edges = X1.shape[0]
        self.n_triangles = X2.shape[0]

        self.labels = labels

        # All atributes that will later not be converted to sparse
        self.tensor_attributes = []
        self.tensor_attributes.extend(list(self.__dict__.keys()))

    def convert_to_sparse(self):
        for instance_var in self.__dict__.keys():
            if instance_var not in self.tensor_attributes:
                instance = getattr(self, instance_var)
                if isinstance(instance, dict):
                    new_value = tensor_to_sparse(instance)
                    setattr(self, instance_var, new_value)

    def unpack_features(self):
        return self.X0, self.X1, self.X2

    def set_features(self, X0, X1, X2):
        self.XO, self.X1, self.X2 = X0, X1, X2

    def to_device(self, DEVICE='cpu'):
        for instance_var in self.__dict__.keys():
            if torch.is_tensor(getattr(self, instance_var)):
                var = getattr(self, instance_var).to(DEVICE)
                setattr(self, instance_var, var)


class SimplicialComplex(simplicial_data):
    """
        Extends the simplicial_data class with specific methods for simplicial complexes,
        including computing Laplacians.

        Args:
            X0 (Tensor): Node features.
            X1 (Tensor): Edge features.
            X2 (Tensor): Triangle features.
            B1 (Tensor): Lower incidence matrix for edge flows.
            B2 (Tensor): Upper incidence matrix for edge flows.
            labels (Tensor): Labels
    """
    def __init__(self, X0, X1, X2, B1, B2, labels):
        super().__init__(X0=X0, X1=X1, X2=X2, labels=labels)

        self.B1 = ensure_input_is_tensor(B1)
        self.B2 = ensure_input_is_tensor(B2)

    def compute_hodge_laplacians(self):
        B1 = tensor_to_sparse(self.B1)
        B2 = tensor_to_sparse(self.B2)

        L0 = B1.mm(torch.transpose(B1, 0, 1))
        self.L0 = sparse_to_tensor(L0)

        L1_upper = B2.mm(torch.transpose(B2, 0, 1))
        L1_lower = torch.transpose(B1, 0, 1).mm(B1)
        self.L1 = sparse_to_tensor(L1_upper + L1_lower)

        L2 = torch.transpose(B2, 0, 1).mm(B2)
        self.L2 = sparse_to_tensor(L2)  # L2 is only created with lower neighbours

    def add_train_test_val_mask(self, train_mask, validation_mask=None, test_mask=None):
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.val_mask = validation_mask

    def get_masks(self):
        return self.train_mask, self.val_mask, self.test_mask

    def get_index_split_dict(self):
        indices = torch.arange(len(self.labels))
        return {
            'train': indices[self.train_mask],
            'valid': indices[self.val_mask],
            'test': indices[self.test_mask]
        }

    def get_label_split(self):
        return self.labels[self.train_mask], self.labels[self.val_mask], self.labels[self.test_mask]

    def unpack_laplacians(self):
        return self.L0, self.L1, self.L2
