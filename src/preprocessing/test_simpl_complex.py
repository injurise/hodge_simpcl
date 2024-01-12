import torch
import networkx as nx
import unittest
from simpl_complex import SimplicialComplex
from src.utils import edge_to_node_matrix, triangle_to_edge_matrix


class TestSIMPCOMP(unittest.TestCase):
    def test_simplicial_complex_basic(self):
        # Create example graph
        G1 = nx.Graph()
        G1.add_nodes_from([1, 2, 3, 4, 5])
        G1.add_edges_from(
            [(1, 5), (1, 2), (2, 3), (3, 1), (3, 4), (4, 5)])  # Adding edges to G automatically orders them.
        # and it doesn't keep track of direction
        trianglesG1 = [list(sorted(x)) for x in nx.enumerate_all_cliques(G1) if len(x) == 3]
        X_0 = torch.tensor([[3], [4], [5], [3], [5]])
        X_1 = torch.tensor([[2], [2], [2], [2], [2], [2]])
        X_2 = torch.tensor([[3]])

        B1 = edge_to_node_matrix(G1.edges, G1.nodes, one_indexed=True)
        B2 = triangle_to_edge_matrix(trianglesG1, G1.edges)

        test_sc = SimplicialComplex(X_0, X_1, X_2, B1, B2, None)
        test_sc.compute_hodge_laplacians()
        test_sc.compute_up_tri_adjacency_matrices()

        test_adj_1_tri = test_sc.ADJ_1_u_tri.to_dense() + test_sc.ADJ_1_l_tri.to_dense()
        # Calculated True one by hand
        true_adj_1_tri = torch.tensor([[0, 1, 1, 0, 0, 1],
                                       [0, 0, 2, 2, 0, 0],
                                       [0, 0, 0, 2, 1, 0],
                                       [0, 0, 0, 0, 1, 0],
                                       [0, 0, 0, 0, 0, 1],
                                       [0, 0, 0, 0, 0, 0]])

        comparison = torch.all(torch.eq(test_adj_1_tri, true_adj_1_tri))
        self.assertEqual(comparison, True)

        #def test_simplicial_complex_basic_zero_indexed(self):
        # Write a test that check if everything works properly when nodes are 0 indexed

    def test_simplicial_complex_triang(self):
        # Write a test with a graph that contains a few triangles and check the variables
        pass
