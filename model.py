import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn
from torch_geometric.data import Batch


# --- 1. GCN Layer  ---
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.batch_norm = batch_norm
        self.residual = residual
        
        # Residual only if dimensions match
        if in_dim != out_dim:
            self.residual = False
        
        # Batch normalization
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        
        # Activation function
        self.activation = activation
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        
        self.conv = geom_nn.GCNConv(in_dim, out_dim, add_self_loops=True, normalize=True)
    
    def forward(self, h, edge_index):
        h_in = h  # for residual connection
        
        # Graph convolution
        h = self.conv(h, edge_index)
        
        # Batch normalization
        if self.batch_norm:
            h = self.batchnorm_h(h)
        
        # Activation function
        if self.activation:
            h = self.activation(h)
        
        # Residual connection
        if self.residual:
            h = h_in + h
        
        # Dropout
        h = self.dropout(h)
        
        return h
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.residual)


# --- 2. Full GCN Network (Matching DGL Benchmark Structure) ---
class GCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        
        # Activation function (ReLU for GCN)
        self.activation = F.relu
        
        # Input dropout
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        # List of GCN layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(GCNLayer(
            in_dim, 
            hidden_dim, 
            self.activation,
            dropout, 
            self.batch_norm, 
            residual=False  # First layer has no residual
        ))
        
        # Hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(GCNLayer(
                hidden_dim,
                hidden_dim,
                self.activation,
                dropout,
                self.batch_norm,
                residual=self.residual
            ))
        
        # Output layer
        self.layers.append(GCNLayer(
            hidden_dim,
            out_dim,
            self.activation,
            dropout,
            self.batch_norm,
            residual=self.residual
        ))
        
        # MLP for classification (readout)
        self.MLP_layer = nn.Linear(out_dim, n_classes)
    
    def forward(self, g, h, e):
        # Handle PyG Batch input
        if isinstance(g, Batch):
            edge_index = g.edge_index
            batch = g.batch
        else:
            # Assume g is edge_index, need batch vector
            raise ValueError("Expected PyG Batch object")
        
        # Input dropout
        h = self.in_feat_dropout(h)
        
        # GCN layers
        for conv in self.layers:
            h = conv(h, edge_index)
        
        # Pooling (readout)
        if self.readout == 'mean':
            h = geom_nn.global_mean_pool(h, batch)
        elif self.readout == 'max':
            h = geom_nn.global_max_pool(h, batch)
        elif self.readout == 'sum':
            h = geom_nn.global_add_pool(h, batch)
        else:
            raise ValueError(f"Unknown readout: {self.readout}")
        
        # Final MLP
        out = self.MLP_layer(h)
        
        return out
    
    def loss(self, pred, label):
        """Cross entropy loss"""
        return F.cross_entropy(pred, label)