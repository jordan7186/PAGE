import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool

"""
GNN Models defined by pytorch geometric
First draft by S.-W Kim, further edits by Y.-M Shin, advised by Prof. W.-Y Shin
"""


class GCN_Conv_Sum(torch.nn.Module):
    def __init__(self, dataset, conv=GCNConv, latent_dim=None, device="cuda"):
        if latent_dim is None:
            latent_dim = [8, 8]
        super(GCN_Conv_Sum, self).__init__()
        self.conv = torch.nn.ModuleList()
        self.device = device
        self.last_dim = latent_dim[-1]

        self.conv.append(conv(dataset.num_features, latent_dim[0]))
        for i in range(len(latent_dim) - 1):
            self.conv.append(conv(latent_dim[i], latent_dim[i + 1]))
        self.last_linear = torch.nn.Linear(latent_dim[-1], 1)

    def reset_parameters(self):
        for conv_layer in self.conv:
            conv_layer.reset_parameters()
        self.last_linear.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        if batch is not None:
            batch = batch.to(self.device)
        for conv in self.conv:
            x = conv(x, edge_index).relu()
        self.entire_embs = x
        x = global_add_pool(x, batch)
        self.embs = x
        x = self.last_linear(x)
        return x.sigmoid()
