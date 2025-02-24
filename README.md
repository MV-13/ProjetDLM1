# DEEP LEARNING PROJECT : Marine VIEILLARD, Romain HÛ
This repository contains all our work on implicit neural representation with periodic activation functions.

# EXPERIMENTS
## IMAGE FITTING
This experiment involves fitting an image with a SIREN model and comparing the result and convergence speed with a ReLU/tanh model.

## SOLVING THE POISSON EQUATION
Reconstructing an image based on its gradient, using the Poisson equation. We again make a comparison between SIREN models and ReLU/tanh models.

## SOLVING THE WAVE EQUATION
Learning a neural representation with a SIREN of a 2D-wavefield based on the wave equation, with dirichlet and neuman initial condition constraints. Comparison with ReLU/tanh models.

# PYTHON CODE
- **utils.py** : useful functions for our work, such as dataset classes for experiments, gaussian function, display functions, etc. ;
- **models.py** : SIREN and ReLU model classes ;
- **experiments.py** : contains 3 code sections for our 3 experiments.