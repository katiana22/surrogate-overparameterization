## Table of contents
* [General info](#general-info)
* [Methods](#methods)
* [Application](#application)
* [Contents](#contents)
* [Datasets](#datasets)
* [Get Started](#get-started)

## General info

This Git repository contains codes for the **'On the influence of over-parameterization in manifold based surrogates and deep neural operators'** paper which can be found here: [https://www.sciencedirect.com/science/article/abs/pii/S0021999123001031](https://www.sciencedirect.com/science/article/abs/pii/S0021999123001031) (JPC) or [https://arxiv.org/abs/2203.05071](https://arxiv.org/abs/2203.05071) (arXiv)

Authors: [Katiana Kontolati](https://scholar.google.com/citations?user=n8wtUDYAAAAJ&hl=en&oi=sra), [Somdatta Goswami](https://scholar.google.com/citations?user=GaKrpSkAAAAJ&hl=en&oi=sra), [Michael D. Shields](https://scholar.google.com/citations?user=hc85Ll0AAAAJ&hl=en), [George Em Karniadakis](https://scholar.google.com/citations?user=yZ0-ywkAAAAJ&hl=en)

## Methods

* **Manifold PCE** or **mPCE** approximates mappings via the identification of low-dimensional embeddings of input and output functions and the construction of a polynomial-based surrogate.

<p align="center">
  <img src="schematics/mPCE-schematic.png" width="700" />
</p>

* **DeepONet** is a deep neural operator model which allows the construction of mapping between infinite dimensional functions via the use of deep neural networks (DNNs).

<p align="center">
  <img src="schematics/DeepONet-schematic.png" width="700" />
</p>

## Application

* The **Brusselator** diffusion-reaction dynamical system is studied, which describes an autocatalytic chemical reaction between two substances. 
* The objective is to approximate the mapping between high-dimensional stochastic initial fields and the evolution of the system across time and space (first row below). The model response is learned for two dynamical states, when the system reaches a **fixed point** in the phase space (second row) and when it reaches a **limit cycle** (third row). 
* We explore the capabilities of the studied models and test them for various regression tasks including their **extrapolation/generalization** ability (performance to out-of-distribution data), **robustness to noise**, ability to handle complex stochastic inputs and **highly nonlinear** mappings.

<p align="center">
  <img src="schematics/Application-schematic.png" width="700" />
</p>

## Contents

* ```data``` - contains files with the input random field data used to generate the train and test data of the model

* ```utils``` - contains python scripts necessary for implementing the surrogate modeling tasks (loading data, plotting etc.)

* ```main_{}.py```/ ```main_{}.ipynb``` - contains python scripts and notebooks for implementing the proposed approaches

## Datasets

To generate the train/test data for the Brusselator model simply run the ```generate_data.py``` script. This will save the generated dataset inside folder ```data/```. The [py-pde](https://github.com/zwicker-group/py-pde) package is used for solving the model.

## Get Started

To clone and use this repository, run the following terminal commands:

**1.** Create an Anaconda Python 3.7 virtual environment:
```
conda create -n surrogate-overparam python==3.7
conda activate surrogate-overparam
```

**2.** Clone the repo:

To clone and use this repository, run the following terminal commands:

```
git clone https://github.com/katiana22/surrogate-overparameterization.git
```
**3.** Install dependencies:

```
cd surrogate-overparameterization
pip install -r requirements.txt
```

## Citation

If you find this GitHub repository useful for your work, please consider citing this work:

```
@article{kontolati2023influence,
  title={On the influence of over-parameterization in manifold based surrogates and deep neural operators},
  author={Kontolati, Katiana and Goswami, Somdatta and Shields, Michael D and Karniadakis, George Em},
  journal={Journal of Computational Physics},
  volume={479},
  pages={112008},
  year={2023},
  publisher={Elsevier}
}
```
______________________

### Contact
For more information or questions please contact us at:   
* kontolati@jhu.edu   
* somdatta_goswami@brown.edu
