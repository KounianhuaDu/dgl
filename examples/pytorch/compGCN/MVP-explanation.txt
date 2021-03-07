# MVP (Minimum Value Product) of DGL GNN Model Example Development

This project is designed for an MVP for GNN model development, and a tutorial for new interns who will implement
GNN-related research papers with DGL.

The GNN model used in this MVP comes from the paper of [CompositionGCN](https://arxiv.org/abs/1911.03082), and was simplified
for the education purpose. In this project, it is called simplified CompGCN.

**NOTE**: The MVP example is to show DGL's standard way of implementing models not for reproducing
any meaningful results.

The Graph Data to Tackle
--------------------------
- Graph data (g): the DGL's built-in CoraGraphDataset.

The Model and Algorithm
-------------------------

For the simplified CompGCN, the detailed math formulas are the same as the original CompGCN paper, except that we did not
use basis vector to represent the relation embedding because in the Cora dataset, there is only one relation. Therefore
we used a learnable parameter as this relation's embedding with the same dimension as the input feature.

The Structure of Example Files
--------------------------------
- **main.py:** The python file for training the simplified CompGCN model with a full graph.
  
- **utils.py:** helper file, providing the circular correlation computation function.
