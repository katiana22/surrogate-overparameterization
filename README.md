## Table of contents
* [General info](#general-info)
* [Methods](#methods)
* [Application](#application)
* [Contents](#contents)
* [Clone](#clone)

## General info

This Git repository contains codes for the new developments and comparisons of manifold-PCE with DeepONet for the construction of surrogate models in nonlinear PDE problems.

## Methods

* _Manifold PCE_ or _mPCE_ approximates mappings via the identification of low-dimensional embeddings of input and output functions and the construction of a polynomial-based surrogate.

<p align="center">
  <img src="schematics/mPCE-schematic.png" width="700" />
</p>

* _DeepONet_ is a deep neural operator model which allows the construction of mapping between infinite dimensional functions via the use of deep neural networks (DNNs).

<p align="center">
  <img src="schematics/DeepONet-schematic.png" width="700" />
</p>

## Application

The _Brusselator_ diffusion-reaction dynamical system is studied, which describes an autocatalytic chemical reaction between two substances. We explore the capabilities of the studied models and test them for various regression tasks including their extrapolation/generalization ability (performance to out-of-distribution data), robustness to noise, ability to handle complex stochastic inputs and highly nonlinear mappings.

<p align="center">
  <img src="schematics/Application-schematic.png" width="700" />
</p>

## Contents

* _scripts_ - contains scripts used for implementing the surrogate modeling tasks

* _mPCE_ - contains notebook for constructing the mPCE model

## Clone

To clone and use this repository, run the following terminal commands.

```
git clone https://github.com/katiana22/mPCE-vs-DeepONet.git
cd mPCE-vs-DeepONet
pip install -r requirements.txt
```

## Citation

If you find this GitHub repository useful for your work, please consider citing this work:

```
ArXiv reference
```
______________________

### Contact
For more information or questions please contact us at:   
* kontolati@jhu.edu   
* somdatta_goswami@brown.edu
