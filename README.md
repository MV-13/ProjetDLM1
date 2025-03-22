# DEEP LEARNING PROJECT : Marine VIEILLARD, Romain HÛ
This repository contains all our work on implicit neural representation with periodic activation functions.


# EXPERIMENTS
## IMAGE FITTING
This experiment involves fitting an image with a SIREN model and comparing the result and convergence speed with a ReLU/tanh model. The user can choose from a list of image from 
skimage.data: camera, cat, astronaut, immunohistochemistry, brick, coffee, rocket.

Each model receives as input a 2D-grid whose coordinates are normalized in [-1, 1] and predicts pixel values for each coordinate.

Models are supervised using the MSE between their predictions and the real pixel values from the chosen image.


## SOLVING THE POISSON EQUATION
Reconstructing an image based on its gradient, using the Poisson equation. We again make a comparison between SIREN models and ReLU/tanh models.

The setting is the same as for image fitting, except that models are supervised using the MSE between the gradient of their predictions and the gradient of the real pixel values, thus solving the Poisson equation.


## SOLVING THE WAVE EQUATION
Learning a neural representation with a SIREN of a 2D-wavefield based on the wave equation, with dirichlet and neuman initial condition constraints.

The model receives as input 2 space coordinates sampled in [-1, 1] and a time coordinate and outputs the value of the wave function at that space-time coordinate.

The model is supervised using a custom loss function to represent the wave equation and the dirichlet and neumann initial conditions. The point source is model as a narrow 2D-gaussian function.

Time coordinates start at 0 and are slowly increase through time : this allows the model to learn initial conditions first and then to carry them over to the remaining training.

Unfortunately, this experiment was not fully concluded. The code runs correctly but seems to fail to converge properly.


## IMPLICIT INPAINTING
Reconstructing a full image based on a masked version. We again make a comparison between SIREN models and ReLU/tanh models.

The setting is the same as for the image fitting task except that models are supervised using the MSE between their masked prediction and the masked image: in other words, the MSE is calculated only on known pixels and pixels reconstructed by the models are ignored.


# PYTHON CODE
- **utils.py**: useful functions for our work, such as dataset classes for experiments, gaussian function, display functions, etc. ;
- **models.py**: sine layer, SIREN and ReLU model classes ;
- **loss_functions.py**: contains loss functions for our different tasks ;
- **differential_operators.py**: contains operators such as gradient, divergence, laplacian and jacobian.
- **experiments.py**: contains 4 ready-to-be-ran code sections for our 4 experiments ;
- **stramlit_utils.py**: a few shortcut functions for streamlit ;
- **app.py**:streamlit app containing 3 sections for the image fitting, Poisson equation and inpainting tasks. Two models are trained simulyaneously for each task : a SIREN and a more standard network. The user is free to choose:
    - model parameters such as number of hidden layers and hidden neurons, learning rates, number of epochs, proportion of masked pixels, etc ;
    - the image to process from the following list : camera, cat, astronaut, immunohistochemistry, brick, coffee, rocket ; 
    - the activation function for the standard model: ReLU, Sigmoid, Tanh, LeakyReLU.
Both models are trained and display their progress in real time allowing a quick comparison of their performance.