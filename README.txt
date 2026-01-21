# Multi-Targeted-Graph-Backdoor-Attack
This repository implements a multi-target backdoor attack on Graph Neural Network for graph classification tasks

## Project Structure
├── main.py                 # Backdoor attack training pipeline

├── clean_model_train.py    # Clean model training (baseline)

├── model.py                # GNN model architecture

├── train_utils.py          # Training and evaluation functions

├── trigger_injection.py    # Trigger generation and injection module

└── README.md


## Requirements
PyTorch 
PyTorch Geometric
NetworkX
NumPy
Matplotlib
scikit-learn

## Instruction for running the codes:

1. Download all of the pyton files in one directory or clone the github repository. You can change the dataset from CIFAR10 Superpixel to other used in the paper mentioned above.

2. Install all the required packages.

3. Run the clean_model_train.py for getting the clean model accuracy obtained by training the GNN model with clean dataset.

4. Run the main.py to inject the trigger into to dataset and train the backdoored GNN model. You will get inndividual ASR for different target classes.
