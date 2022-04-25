# Multi-scale Physical Representations for Approximating PDE Solutions with Graph Neural Operators

This repository contains the code for the paper:

- [Multi-scale Physical Representations for Approximating PDE Solutions with Graph Neural Operators](https://openreview.net/forum?id=rx9TVZJax5)

In this work, we build upon Multipole Graph Neural Operator (MGNO) and numerical schemes to create novel architectures to tackle multi-scale physical representation problem.

## Requirements

- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)

## Files

- MGKN_orthogonal_burgers1d.py is the Multipole Graph Neural Operator (MGKN) file for the 1D Burgers equation.
- GKN_orthogonal_burgers1d.py is the  Graph Kernel Network (GKN) file for the 1D Burgers equation.
- MLP_GCN_orthogonal_burgers1d.py is the multilayer perceptron (MLP) and graph convolutional network (GCN) file for the 1D Burgers equation.
- MGKN_orthogonal_darcy2d.py is the Multipole Graph Neural Operator (MGKN) file for the 2D Darcy flow equation.
- GKN_orthogonal_darcy2d.py is the  Graph Kernel Network (GKN) file for the 2D Darcy flow equation.
- MLP_GCN_orthogonal_darcy2d.py is the multilayer perceptron (MLP) and graph convolutional network (GCN) file for the 2D Darcy flow equation.
- src contains the code for the graph data creation, the different models and some utilities files from the original FNO code.

## Datasets

We used the Burgers equation and Darcy flow equation datasets from the GNO paper. The data generation configuration can be found in their paper.

- [PDE datasets](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-)

See [github](https://github.com/zongyi-li/fourier_neural_operator) for addtional information.

### Usage

```bash
python3 MGKN_orthogonal_burgers1d.py
```

### Acknowledgement

This code is adapted from [this repository](https://github.com/zongyi-li/graph-pde) by Zongyi Li.
