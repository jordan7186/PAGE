from torch_geometric.data import Data
import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from PrototypeDiscovery import PrototypeDiscovery
from tqdm import tqdm

"""
GNN Models defined by pytorch geometric
First draft by S.-W Kim, further edits by Y.-M Shin, advised by Prof. W.-Y Shin
Additional contributions: E.-B Yun
"""


class prototype_explanation:
    def __init__(
        self,
        gnn_model,
        dataset,
        data_name,
        n_components,
        with_hydrogen=True,
        incomplete_experiment=False,
    ):

        self.model = gnn_model
        self.data = dataset
        self.data_type = data_name
        self.n_components = n_components
        self.with_hydrogen = with_hydrogen
        self.incomplete_experiment = incomplete_experiment

        if self.incomplete_experiment is False:
            self.GMM()

    def multiple_mahalanobis(self, x, means, prec):
        return np.diag(((x - means).dot(prec)).dot((x - means).transpose()))

    def shifted_graph_to_torch(self, G, features):
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

    def visualizing(
        self,
        torch_data,
        figure_size=(8, 6),
        title="Resulting Graph",
    ):
        """
        :param G: torch_geometric type data
        :param title: title of the plots
        :return: graph figure
        """
        plt.figure(figsize=figure_size)
        node_colors = []
        node_size = []
        edge_size = []
        G = nx.Graph()

        for i in range(torch_data.x.shape[0]):
            G.add_node(i)
            if torch.argmax(torch_data.x[i, :]) == 0:
                node_colors.append("red")
            elif torch.argmax(torch_data.x[i, :]) == 1:
                node_colors.append("lime")
            else:
                node_colors.append("orange")
            node_size.append(800)

        for s in range(torch_data.edge_index.shape[1]):
            G.add_edge(
                torch_data.edge_index[0, s].item(),
                torch_data.edge_index[1, s].item(),
            )
            edge_size.append(1)
        nx.draw(G, node_color=node_colors, node_size=node_size, width=edge_size)
        plt.title(title)
        plt.show()

    def GMM(self):
        assert self.incomplete_experiment is False
        """
        :param label: Which label(class) to be explained
        :param k: Number of nearest neighbors from the centroid
        :param cluster_index : A Cluster index which we want to see
        :return: list of the nearest nodes
        """
        self.model.eval()
        with torch.no_grad():
            self.model(self.data[0])
        embeddings = torch.empty((1, self.model.embs.shape[1]))
        ys = []
        for part_d in self.data:
            with torch.no_grad():
                _ = self.model(part_d)
            embeddings = torch.vstack([embeddings, self.model.embs])
            ys.append(part_d.y.item())
        entire_embs = embeddings[1:, :].detach().numpy()
        self.y1 = np.where(np.array(ys) == 1.0)[0]
        self.y0 = np.where(np.array(ys) == 0.0)[0]
        self.emb1 = entire_embs[self.y1, :]
        self.emb0 = entire_embs[self.y0, :]
        self.clus1 = GaussianMixture(
            n_components=self.n_components, random_state=0
        ).fit(self.emb1)
        self.clus0 = GaussianMixture(
            n_components=self.n_components, random_state=0
        ).fit(self.emb0)

    def generate_nearest_nodes(self, label, cluster_index, k=3):
        assert self.incomplete_experiment is False

        if cluster_index >= self.n_components:
            raise TypeError(
                "Cluster index should be given smaller integer than number of components"
            )

        if label == 0:
            pos = self.get_position_for_label_0(cluster_index, k)
        elif label == 1:
            pos = self.get_position_for_label_1(cluster_index, k)
        else:
            raise TypeError("Label should be given either 1 or 0")
        return pos

    def get_position_for_label_1(self, cluster_index, k):
        dists = self.multiple_mahalanobis(
            self.emb1,
            self.clus1.means_[cluster_index],
            self.clus1.precisions_[cluster_index],
        )
        sorting = np.sort(dists)
        top_nodes = []
        index_counting = 0
        while len(top_nodes) < k:
            index_counting += 1
            if sorting[int(index_counting - 1)] not in top_nodes:
                top_nodes.append(sorting[int(index_counting - 1)])
        return [self.y1[np.where(dists == top_nodes[i])[0]][0] for i in range(k)]

    def get_position_for_label_0(self, cluster_index, k):
        dists = self.multiple_mahalanobis(
            self.emb0,
            self.clus0.means_[cluster_index],
            self.clus0.precisions_[cluster_index],
        )
        sorting = np.sort(dists)
        top_nodes = []
        index_counting = 0
        while len(top_nodes) < k:
            index_counting += 1
            if sorting[int(index_counting - 1)] not in top_nodes:
                top_nodes.append(sorting[int(index_counting - 1)])
        return [self.y0[np.where(dists == top_nodes[i])[0]][0] for i in range(k)]

    def generate_prototype(
        self,
        label,
        cluster_index,
        max_epochs=1000,
        max_node=5,
        show_figure=False,
        given_title="Resulting Graph",
        verbose=0,
    ):
        pos = self.generate_nearest_nodes(label=label, cluster_index=cluster_index)
        i = 0
        min_prob = 0.9
        decay = 10
        FLAG = False

        while i < 50:  # Keep iteration until we get limit probability
            if verbose:
                print("Iteration / Order: ", i)

            i += 1  # Adding Order one by one
            MCS_generator = PrototypeDiscovery(
                self.model,
                G1=self.data[int(pos[0])],
                G2=self.data[int(pos[1])],
                G3=self.data[int(pos[2])],
                data_type=self.data_type,
                label=label,
                with_hydrogen=self.with_hydrogen,
            )

            proto_G = MCS_generator.subgraph_finding(
                max_epochs=max_epochs,
                decay=decay,
                order=i,
                max_node=max_node,
                verbose=verbose,
            )

            if proto_G == -1:  # Failed in searching MCS -> Move on to next order
                if verbose:
                    print("MCS not found")
                continue  # Do once again

            if verbose:
                print("Assessing prototype...")

            with torch.no_grad():
                prob = self.model(proto_G)

            if label == 0:
                if prob < (1 - min_prob):
                    if verbose:
                        print("Found: probability ", 1 - prob)
                    FLAG = True
                    break
                else:
                    if verbose:
                        self.visualizing(
                            proto_G, data_type=self.data_type, title=given_title
                        )
                        print("Not found: probability ", 1 - prob)
            elif prob > min_prob:
                if verbose:
                    print("Found: probability ", prob)
                FLAG = True
                break
            else:
                if verbose:
                    self.visualizing(
                        proto_G, data_type=self.data_type, title=given_title
                    )
                    print("Not found: probability ", prob)

        if show_figure:
            self.visualizing(proto_G, data_type=self.data_type, title=given_title)
        if FLAG:
            return proto_G
        print("No prototype found")
        return -1

    def generate_prototype_quant(
        self,
        label: int,
        cluster_index: int,
        budget: int = 5,
        max_epochs: int = 1000,
        max_node: int = 5,
        verbose: int = 0,
        print_index: bool = False,
        explicit_position: list = None,
    ):
        if explicit_position is None:
            pos = self.generate_nearest_nodes(label=label, cluster_index=cluster_index)
        elif type(explicit_position) == list:
            assert self.incomplete_experiment
            pos = explicit_position
        else:
            raise TypeError(
                "Either set explicit_position as None or give index as list of int"
            )
        start = 0
        prob_log = []
        proto_log = []
        for start in tqdm(range(budget), desc="Prototype budget"):
            if verbose:
                print(f"Order: {start}")

            start += 1  # Adding Order one by one
            MCS_generator = PrototypeDiscovery(
                self.model,
                G1=self.data[int(pos[0])],
                G2=self.data[int(pos[1])],
                G3=self.data[int(pos[2])],
                data_type=self.data_type,
                label=label,
                with_hydrogen=self.with_hydrogen,
            )

            proto_G = MCS_generator.subgraph_finding(
                max_epochs=max_epochs,
                decay=10,
                order=start,
                max_node=max_node,
                verbose=verbose,
            )

            proto_log.append(proto_G)

            if proto_G == -1:  # Failed in searching MCS -> Move on to next order
                if verbose:
                    print("MCS not found")
                continue  # Do next iteration

            if verbose:
                print("Assessing prototype...")

            with torch.no_grad():
                prob = self.model(proto_G)

            prob = 1 - prob if label == 0 else prob
            prob_log.append(prob)

        max_prob_index = prob_log.index(max(prob_log))
        clean_list = [round(num.item(), 4) for num in prob_log]
        return_G = proto_log[max_prob_index]

        if print_index:
            print(f"Return index: {max_prob_index}")
            print(f"Return probability: {clean_list}")

        if return_G == -1:
            raise ValueError

        return return_G
