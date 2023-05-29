import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,GINEConv,HEATConv,RGCNConv,FiLMConv
import torch.nn as nn
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Dataset,DataLoader,HeteroData
from torch_geometric.nn import global_mean_pool
class RecGraph(torch.nn.Module):
    def __init__(self,num_items,embedding_dim=128,layer=3,add_self_loops=False):
        super().__init__()
        self.num_items = num_items
        self.item_emb = nn.Embedding(self.num_items,embedding_dim)
        nn.init.normal_(self.item_emb.weight,std=0.1)
        self.graph = RGCNConv(128,128,2,num_bases=30)
        self.linear = nn.Linear(128,num_items)
        self.gcn = RGCNConv(128,num_items,2)
    def forward(self,data):
        x = self.item_emb(torch.LongTensor(range(self.num_items)).cuda())
        graph = self.graph(x,data.edges,data.edge_type)
        graph = global_mean_pool(graph, torch.zeros_like(torch.LongTensor([graph.shape[0]])).cuda())
        graph = self.linear(graph)
        return graph
if __name__ == '__main__':
    model = RecGraph(207520)
    print(model)


