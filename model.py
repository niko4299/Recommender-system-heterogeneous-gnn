import torch
import torch_geometric.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.hetero_conv import HeteroConv

class HeteroGATConvGNN(torch.nn.Module):
    def __init__(self, metadata,args):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(args.num_layers):
            conv = HeteroConv({
                edge_type: nn.GATv2Conv(-1, args.hidden_channels,dropout = args.dropout if i < args.num_layers-1 else 0)
                for edge_type in metadata[1]
            })
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.selu(x) for key, x in x_dict.items()}
        
        return x_dict
    
    def decode(self,x_dict, edge_label_index):
        src, dst = x_dict['user'][edge_label_index[0]], x_dict['movie'][edge_label_index[1]]

        return (src * dst).sum(dim = 1)