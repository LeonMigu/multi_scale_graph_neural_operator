# Multi-scale Physical Representations for Approximating PDE Solutions with Graph Neural Operators

This repository contains the code for the paper:

- [Multi-scale Physical Representations for Approximating PDE Solutions with Graph Neural Operators]([https://openreview.net/forum?id=rx9TVZJax5](https://proceedings.mlr.press/v196/migus22a.html))

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

## References
Leon Migus, Yuan Yin, Jocelyn Ahmed Mazari, Patrick Gallinari Proceedings of Topological, Algebraic, and Geometric Learning Workshops 2022, PMLR 196:332-340, 2022. 
```

@InProceedings{pmlr-v196-migus22a,
  title = 	 {Multi-Scale Physical Representations for Approximating PDE Solutions with Graph Neural Operators},
  author =       {Migus, Leon and Yin, Yuan and Ahmed Mazari, Jocelyn and  Gallinari, Patrick},
  booktitle = 	 {Proceedings of Topological, Algebraic, and Geometric Learning Workshops 2022},
  pages = 	 {332--340},
  year = 	 {2022},
  editor = 	 {Cloninger, Alexander and Doster, Timothy and Emerson, Tegan and Kaul, Manohar and Ktena, Ira and Kvinge, Henry and Miolane, Nina and Rieck, Bastian and Tymochko, Sarah and Wolf, Guy},
  volume = 	 {196},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {25 Feb--22 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v196/migus22a/migus22a.pdf},
  url = 	 {https://proceedings.mlr.press/v196/migus22a.html},
  abstract = 	 {Representing physical signals at different scales is among the most challenging problems in engineering. Several multi-scale modeling tools have been developed to describe physical systems governed by Partial Differential Equations (PDEs). These tools are at the crossroad of principled physical models and numerical schema. Recently, data-driven models have been introduced to speed-up the approximation of PDE solutions compared to numerical solvers. Among these recent data-driven methods, neural integral operators are a class that learn a mapping between function spaces. These functions are discretized on graphs (meshes) which are appropriate for modeling interactions in physical phenomena. In this work, we study three multi-resolution schema with integral kernel operators that can be approximated with Message Passing Graph Neural Networks (MPGNNs). To validate our study, we make extensive MPGNNs experiments with well-chosen metrics considering steady and unsteady PDEs.}
}
```
