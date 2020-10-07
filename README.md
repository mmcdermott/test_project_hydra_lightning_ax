# Test Project: Experiment Configuration & Hyperparameter Tuning
In this test project, I'll be exploring a few tools:
  1. [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
  2. [Botorch/Ax](https://botorch.org/docs/botorch_and_ax)
  3. [Hydra](https://hydra.cc/)

# Iteration v1: MNIST Classification
For iteration 1 of this project, I'll work on the classic problem; MNIST classification. Ultimately, I want to
extend this to some more personally interesting use-cases, but for now I'll stick just with MNIST
classification.

In the interest of letting this repo serve as multiple investigation points, I'll actually make a bunch of
subdirectories for various iterations; this v1 will live in `v1_MNIST`.

# Extensions:
  1. PyTorch Bolts
  2. Use https://github.com/PyTorchLightning/deep-learning-project-template

# Log
##### 09/20/2020
  1. Set up project
  2. Built basic MNIST classifier using PyTorch Lightning
  
##### 10/07/2020
  1. Refined MNIST classifier (added validation step and basic metrics).
  2. Began exploring hyperparameter search options. Ax seems to be missing some nice features, and is surprisingly poorly documented. Pro-tip: When googling for things related to Ax, include "Facebook" in the search query, or you get unrelated results.
