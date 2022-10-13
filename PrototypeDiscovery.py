import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_networkx, subgraph
import networkx as nx
import numpy as np
from tqdm import tqdm

"""
GNN Models defined by pytorch geometric
First draft by S.-W Kim, further edits by Y.-M Shin, advised by Prof. W.-Y Shin
Additional contributions: E.-B Yun
"""


class PrototypeDiscovery:
    def __init__(self, model, G1, G2, G3, data_type, with_hydrogen=True, label=1):
        self.model = model
        self.data_type = data_type
        self.with_hydrogen = with_hydrogen
        if label in (0, 1):
            self.label = label
        else:
            raise ValueError("Binary Classification task")

        self.G1 = G1
        self.G2 = G2
        self.G3 = G3

        self.V1 = G1.x.shape[0]
        self.V2 = G2.x.shape[0]
        self.V3 = G3.x.shape[0]

        self.A1 = to_dense_adj(self.G1.edge_index)[0]
        self.A2 = to_dense_adj(self.G2.edge_index)[0]
        self.A3 = to_dense_adj(self.G3.edge_index)[0]
        self.Y = self.matching_matrix_generation()

    def get_U(self, G):  # G : torch.geometric  data
        self.model(G)
        return self.model.entire_embs

    def shifted_graph_to_torch(self, G, features):  # Re-Indexing nodes
        node_trans = {}
        edge_index = torch.tensor([[0, 0]], dtype=torch.long)
        num_feat = 3
        feature_vector = torch.zeros((1, num_feat), dtype=torch.float)
        for i, j in enumerate(G.nodes):
            node_trans[j] = i
            added_feat = torch.zeros((1, num_feat), dtype=torch.float)
            added_feat[0, features[i]] = 1
            feature_vector = torch.vstack([feature_vector, added_feat])
        for n1, n2 in G.edges:
            added_edge = torch.tensor(
                [[node_trans[n1], node_trans[n2]], [node_trans[n2], node_trans[n1]]],
                dtype=torch.long,
            )
            edge_index = torch.vstack([edge_index, added_edge])
        feature_vector = feature_vector[1:, :]
        edge_index = edge_index[1:, :]
        return Data(x=feature_vector, edge_index=edge_index.t().contiguous())

    def label_mask(self):  # Detecting joint node features
        x1 = self.G1.x
        x2 = self.G2.x
        x3 = self.G3.x
        X = torch.zeros((x1.shape[0], x2.shape[0], x3.shape[0]))
        for i in range(x1.shape[0]):
            for j in range(x2.shape[0]):
                for k in range(x3.shape[0]):
                    X[i, j, k] = torch.sum(x1[i] * x2[j] * x3[k])
        return X

    def matching_matrix_generation(self):
        self.U1 = self.get_U(self.G1)
        self.U2 = self.get_U(self.G2)
        self.U3 = self.get_U(self.G3)
        X = torch.zeros((self.V1, self.V2, self.V3))
        M = torch.max(X)

        for i in range(self.U1.shape[0]):
            u1_i = self.U1[i]
            for j in range(self.U2.shape[0]):
                u2_j = self.U2[j]
                for k in range(self.U3.shape[0]):
                    u3_k = self.U3[k]
                    X[i, j, k] = torch.sum(u1_i * u2_j * u3_k)

        exp_X = torch.exp(X - M)
        partition_d1 = exp_X.sum(0, keepdim=True) + 1e-5
        partition_d2 = exp_X.sum(1, keepdim=True) + 1e-5
        partition_d3 = exp_X.sum(2, keepdim=True) + 1e-5

        p_dim1 = exp_X / partition_d1 * torch.sigmoid(X.sum(0, keepdim=True) / self.V1)
        p_dim2 = exp_X / partition_d2 * torch.sigmoid(X.sum(1, keepdim=True) / self.V2)
        p_dim3 = exp_X / partition_d3 * torch.sigmoid(X.sum(2, keepdim=True) / self.V3)

        p = (p_dim1 + p_dim2 + p_dim3) / 3
        return p * self.label_mask()

    def get_index(self, n):
        v1 = n // (self.V2 * self.V3)
        v2 = (n % (self.V2 * self.V3)) // self.V3
        v3 = (n % (self.V2 * self.V3)) % self.V3
        return v1, v2, v3

    def subgraph_finding(
        self, max_epochs=100, order=1, decay=1.5, max_node=5, verbose=0
    ):
        v1, v2, v3 = self.initial_pair(order)

        node1 = [v1]
        node2 = [v2]
        node3 = [v3]
        for _ in (
            tqdm(range(1, 1 + max_epochs)) if verbose else range(1, 1 + max_epochs)
        ):

            neigh1, neigh2, neigh3 = self.find_nbd(v1, v2, v3)  # Find first_neighbors
            if (
                (len(neigh1) == 0) | (len(neigh2) == 0) | (len(neigh3) == 0)
            ):  # Cannot add neighbor anymore
                return -1  ## Failed in searching. Starts from next case

            v1 = neigh1[0]
            v2 = neigh2[0]
            v3 = neigh3[0]

            node1.append(v1)
            node2.append(v2)
            node3.append(v3)

            self.Y[v1, :, :] = self.Y[v1, :, :] / decay
            self.Y[:, v2, :] = self.Y[:, v2, :] / decay
            self.Y[:, :, v3] = self.Y[:, :, v3] / decay

            n1, n2, n3 = len(set(node1)), len(set(node2)), len(set(node3))

            # If certain graph has achieved max nodes, stop adding nodes here.
            if n1 > max_node:
                node1.pop()
            if n2 > max_node:
                node2.pop()
            if n3 > max_node:
                node3.pop()

            if min(n1, n2, n3) > max_node:
                if verbose:
                    print("Max node reached")
                break

        n1, n2, n3 = len(set(node1)), len(set(node2)), len(set(node3))

        tmp_G1 = self.showing_result(tmp_node=node1, orig_graph=self.G1)
        tmp_G2 = self.showing_result(tmp_node=node2, orig_graph=self.G2)
        tmp_G3 = self.showing_result(tmp_node=node3, orig_graph=self.G3)

        with torch.no_grad():
            p1 = self.model(tmp_G1)
            p2 = self.model(tmp_G2)
            p3 = self.model(tmp_G3)

        if self.label == 1:  # Label is 1
            v = np.argmax(
                [
                    p1.to("cpu").detach().item(),
                    p2.to("cpu").detach().item(),
                    p3.to("cpu").detach().item(),
                ]
            )
        else:  # Label is zero
            v = np.argmin(
                [
                    p1.to("cpu").detach().item(),
                    p2.to("cpu").detach().item(),
                    p3.to("cpu").detach().item(),
                ]
            )

        nodes = [node1, node2, node3][int(v)]
        Gs = [self.G1, self.G2, self.G3][int(v)]

        if verbose:
            print("Returning prototype...")

        return self.showing_result(tmp_node=nodes, orig_graph=Gs)

    def showing_result(self, tmp_node, orig_graph):
        partial_edge_index = subgraph(tmp_node, orig_graph.edge_index)[0]
        data = Data(orig_graph.x, partial_edge_index)
        S = to_networkx(data, to_undirected=True)
        index = [i for i, node in enumerate(S.nodes) if node not in nx.isolates(S)]
        S.remove_nodes_from(list(nx.isolates(S)))
        node_feature = torch.where(data.x[index] == 1)[1]
        return self.shifted_graph_to_torch(G=S, features=node_feature)

    def select_subgraph(self, max_index=5, max_epochs=1000, decay=1.5, threshold=0.001):
        sizes = []
        node1s = []
        node2s = []
        for m in range(1, max_index + 1):
            node1, node2 = self.subgraph_finding(
                order=m, max_epochs=max_epochs, decay=decay, threshold=threshold
            )
            l1 = len(np.unique(np.array(node1)))
            l2 = len(np.unique(np.array(node2)))
            node1s.append(node1)
            node2s.append(node2)
            size = min(l1, l2)
            sizes.append(size)
        sizes = np.array(sizes)
        M = np.argmax(sizes)
        node1 = node1s[M]
        node2 = node2s[M]
        return node1, node2

    def initial_pair(self, n):
        Y = self.Y[:, :, :]
        index = torch.argmax(Y)
        v1, v2, v3 = self.get_index(index)

        feature_i = torch.where(self.G1.x[v1] == 1)[0]
        feature_j = torch.where(self.G2.x[v2] == 1)[0]
        feature_k = torch.where(self.G3.x[v3] == 1)[0]

        while not (feature_i == feature_j == feature_k):
            self.Y[v1, v2, v3] = 0
            Y[v1, :, :] = 0
            Y[:, v2, :] = 0
            Y[:, :, v3] = 0
            index = torch.argmax(Y)
            v1, v2, v3 = self.get_index(index)

            feature_i = torch.where(self.G1.x[v1] == 1)[0]
            feature_j = torch.where(self.G2.x[v2] == 1)[0]
            feature_k = torch.where(self.G3.x[v3] == 1)[0]

        v1 = int(v1)
        v2 = int(v2)
        v3 = int(v3)
        if n > 1:
            S = 1
            while S < n:
                feature_i = 1
                feature_j = 2
                feature_k = 3
                while not (feature_i == feature_j == feature_k):
                    Y[v1, :, :] = 0
                    Y[:, v2, :] = 0
                    Y[:, :, v3] = 0
                    index = torch.argmax(Y)
                    v1, v2, v3 = self.get_index(index)

                    feature_i = torch.where(self.G1.x[v1] == 1)[0]
                    feature_j = torch.where(self.G2.x[v2] == 1)[0]
                    feature_k = torch.where(self.G3.x[v3] == 1)[0]

                v1 = int(v1)
                v2 = int(v2)
                v3 = int(v3)
                S += 1

        return v1, v2, v3

    def find_nbd(self, v1, v2, v3):

        a1 = self.A1[v1].reshape(-1, 1)
        nbd1 = torch.where(a1 == 1)[0]

        a2 = self.A2[v2].reshape(-1, 1)
        nbd2 = torch.where(a2 == 1)[0]

        a3 = self.A3[v3].reshape(-1, 1)
        nbd3 = torch.where(a3 == 1)[0]

        ## Efficient Method
        nbd_1 = []
        for i in nbd1:
            nbd_1 += list(np.repeat(i, int(nbd2.shape[0] * nbd3.shape[0])))
        nbd_1 = torch.tensor(nbd_1)

        nbd_2 = []
        for j in nbd2:
            nbd_2 += list(np.repeat(j, int(nbd3.shape[0])))
        nbd_2 = torch.tensor(nbd_2)
        nbd_2 = nbd_2.repeat(nbd1.shape[0])

        nbd_3 = nbd3.repeat(nbd1.shape[0] * nbd2.shape[0])

        B = self.Y[nbd_1, nbd_2, nbd_3]
        C = torch.argsort(B, dim=0, descending=True)

        NB1 = []
        NB2 = []
        NB3 = []

        for index in C:

            i = nbd_1[index]
            j = nbd_2[index]
            k = nbd_3[index]

            feature_i = torch.where(self.G1.x[i] == 1)[0]
            feature_j = torch.where(self.G2.x[j] == 1)[0]
            feature_k = torch.where(self.G3.x[k] == 1)[0]

            if not (feature_i == feature_j == feature_k):
                self.Y[i, j, k] = 0
                continue

            NB1.append(int(i))
            NB2.append(int(j))
            NB3.append(int(k))

        return NB1, NB2, NB3
