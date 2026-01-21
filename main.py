import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import networkx as nx
import pickle
import copy

import sys

import networkx as nx
import numpy as np
import random
import torch
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import matplotlib.pyplot as plt
from math import ceil

from torch_geometric.data import Data
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import GNNBenchmarkDataset
import torch_geometric.nn as geom_nn
from torch_geometric.data import Batch
from torch_geometric.utils import to_undirected
from torch_geometric.loader import DataLoader

# Import from separate modules
from trigger_gen_injection import trigger_gen_injection
from model import GCNLayer, GCNNet
from train_utils import train_epoch, evaluate_network


device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


train_dataset = GNNBenchmarkDataset(root='dataset', name='CIFAR10', split='train')
test_dataset = GNNBenchmarkDataset(root='dataset', name='CIFAR10', split='test')


# For making the data with one node feature

for i in range(len(train_dataset)):
    data = train_dataset[i]
    # to_undirected can handle edge_attr automatically
    data.x = torch.cat([data.x, data.pos], dim=1)
    train_dataset._data_list[i] = data



for i in range(len(test_dataset)):
    data = test_dataset[i]
    # to_undirected can handle edge_attr automatically
    data.x = torch.cat([data.x, data.pos], dim=1)
    test_dataset._data_list[i] = data


test_dataset0 = GNNBenchmarkDataset(root='dataset', name='CIFAR10', split='test')
for i in range(len(test_dataset)):
    data = test_dataset0[i]
    # to_undirected can handle edge_attr automatically
    data.x = torch.cat([data.x, data.pos], dim=1)
    test_dataset0._data_list[i] = data

test_dataset1 = GNNBenchmarkDataset(root='dataset', name='CIFAR10', split='test')
for i in range(len(test_dataset1)):
    data = test_dataset1[i]
    # to_undirected can handle edge_attr automatically
    data.x = torch.cat([data.x, data.pos], dim=1)
    test_dataset1._data_list[i] = data


test_dataset2 = GNNBenchmarkDataset(root='dataset', name='CIFAR10', split='test')
for i in range(len(test_dataset2)):
    data = test_dataset2[i]
    # to_undirected can handle edge_attr automatically
    data.x = torch.cat([data.x, data.pos], dim=1)
    test_dataset2._data_list[i] = data


print(train_dataset[0])

        
    
print(test_dataset2[0])


frac = 0.05
avg_num_nodes = 117
var = 0.2
num_backdoor_nodes = ceil(avg_num_nodes*var)
seed = 0
target_label = [0,1,2]
prob = 0.8
num_trigger = 1
node_feat_size = 5
num_classes = 10


train_backdoor, test_backdoor_label_0, test_backdoor_label_1, test_backdoor_label_2 = trigger_gen_injection(train_dataset, test_dataset0, test_dataset1, test_dataset2, frac, num_backdoor_nodes, seed, target_label, prob, num_trigger, node_feat_size, num_classes)



for g in test_backdoor_label_0:
    g.y = torch.tensor([target_label[0]], dtype=torch.long)


for g in test_backdoor_label_1:
    g.y = torch.tensor([target_label[1]], dtype=torch.long)


for g in test_backdoor_label_2:
    g.y = torch.tensor([target_label[2]], dtype=torch.long)
    

print(test_backdoor_label_1[5].y)
print(test_backdoor_label_2[5].y)


# --- 5. Setup and Training ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    if hasattr(train_dataset[0], 'edge_attr') and train_dataset[0].edge_attr is not None:
        in_dim_edge = train_dataset[0].edge_attr.shape[1] if len(train_dataset[0].edge_attr.shape) > 1 else 1
    else:
        in_dim_edge = 0
except:
    in_dim_edge = 0

net_params = {
    'in_dim': train_dataset[0].x.shape[1],
    'in_dim_edge': in_dim_edge,
    'hidden_dim': 256,
    'out_dim': 256,
    'n_classes': 10,
    'L': 5,  # Number of layers
    'readout': 'mean',
    'in_feat_dropout': 0.0,
    'dropout': 0.0,
    'batch_norm': True,
    'residual': True,
    'device': device,
    'batch_size': 512
}

# Create model
model = GCNNet(net_params).to(device)

# Count parameters
total_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total parameters: {total_param:,}')
print(f'\nModel Configuration:')
print(f'  Layers (L): {net_params["L"]}')
print(f'  Hidden dim: {net_params["hidden_dim"]}')
print(f'  Out dim: {net_params["out_dim"]}')
print(f'  Architecture: {net_params["in_dim"]} → {net_params["hidden_dim"]} → {net_params["hidden_dim"]} → {net_params["out_dim"]}')
print(f'  Final MLP: {net_params["out_dim"]} → {net_params["n_classes"]}')
print(f'  Readout: {net_params["readout"]}')
print(f'  Residual: {net_params["residual"]}')
print(f'  Batch norm: {net_params["batch_norm"]}')
print(f'  Activation: ReLU\n')

# Optimizer and scheduler
params = {
    'init_lr': 0.002,
    'lr_reduce_factor': 0.5,
    'lr_schedule_patience': 10,
    'min_lr': 1e-5,
    'weight_decay': 0.0,
    'batch_size': 512,
    'epochs': 300
}

optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=params['lr_reduce_factor'],
    patience=params['lr_schedule_patience']
)

# Create data loaders with PyG's built-in DataLoader
train_loader = DataLoader(
    train_backdoor, 
    batch_size=params['batch_size'], 
    shuffle=True,
    num_workers=0,      # Set to 4-8 for faster loading if you have multiple CPU cores
    pin_memory=True if torch.cuda.is_available() else False
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=params['batch_size'], 
    shuffle=False,
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)

test_loader_0 = DataLoader(
    test_backdoor_label_0, 
    batch_size=params['batch_size'], 
    shuffle=False,
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)

test_loader_1 = DataLoader(
    test_backdoor_label_1, 
    batch_size=params['batch_size'], 
    shuffle=False,
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)

test_loader_2 = DataLoader(
    test_backdoor_label_2, 
    batch_size=params['batch_size'], 
    shuffle=False,
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)



print(f"Training Graphs: {len(train_backdoor)}")
print(f"Test Graphs: {len(test_dataset)}")
print(f"Test Graphs_0: {len(test_backdoor_label_0)}")
print(f"Test Graphs_1: {len(test_backdoor_label_0)}")
print(f"Test Graphs_2: {len(test_backdoor_label_0)}")
print(f"Batch size: {params['batch_size']}")
print(f"Starting training...\n")

# --- 6. Training Loop ---
epoch_train_losses, epoch_test_losses = [], []
epoch_train_accs, epoch_test_accs = [], []
ASR_0 = []
ASR_1 = []
ASR_2 = []
best_test_acc = 0
best_epoch = 0


for epoch in range(params['epochs']):
    # Train
    train_loss, train_acc = train_epoch(model, optimizer, device, train_loader)
    
    # Evaluate
    test_loss, test_acc = evaluate_network(model, device, test_loader)
    loss_0, acc_0 = evaluate_network(model, device, test_loader_0)
    loss_1, acc_1 = evaluate_network(model, device, test_loader_1)
    loss_2, acc_2 = evaluate_network(model, device, test_loader_2)
    
    # Store metrics
    epoch_train_losses.append(train_loss)
    epoch_test_losses.append(test_loss)
    epoch_train_accs.append(train_acc)
    epoch_test_accs.append(test_acc)
    ASR_0.append(acc_0)
    ASR_1.append(acc_1)
    ASR_2.append(acc_2)
    
    
    # Learning rate scheduling (on test loss - using test as validation in this setup)
    scheduler.step(test_loss)
    
    # Track best
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_epoch = epoch + 1
    
    # Print progress
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch+1:4d}/{params["epochs"]}] | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
              f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'ASR_0 = {acc_0}, ASR_1 = {acc_1}, ASR_2 = {acc_2}')
        
    
    # Early stopping
    if optimizer.param_groups[0]['lr'] < params['min_lr']:
        print(f"\n!! LR reached MIN_LR. Stopping at epoch {epoch+1}")
        break

print(f"\n{'='*80}")
print(f"Training Complete!")
print(f"Best Test Accuracy: {best_test_acc:.4f} at epoch {best_epoch}")
print(f"Final Test Accuracy: {test_acc:.4f}")
print(f"Final Train Accuracy: {train_acc:.4f}")
print(f"{'='*80}\n")

# --- 7. Visualization ---
plt.plot(epoch_test_accs, label='Test Accuracy', linewidth=2)
plt.plot(ASR_0, label='ASR_0', linewidth=2)
plt.plot(ASR_1, label='ASR_1', linewidth=2)
plt.plot(ASR_2, label='ASR_2', linewidth=2)
plt.axhline(y=best_test_acc, color='r', linestyle='--', linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'3 Target Backdoor Attack over GCN model on CIFAR10 Dataset [Trigger_size = {num_backdoor_nodes} ({var*100})% ]')
plt.legend()

plt.show()