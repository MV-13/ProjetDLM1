import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import utils
import models
import differential_operators as diff


# Getting device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#####################################################################################################
# EXPERIMENT 1 : IMAGE FITTING
#####################################################################################################

cameraman = utils.ImageFitting(256)
dataloader = DataLoader(cameraman, batch_size = 1, pin_memory = True, num_workers = 0)
X, y = next(iter(dataloader))
X, y = X.to(device), y.to(device)

siren = models.Siren(in_features = 2, out_features = 1, hidden_features = 256,
                     hidden_layers = 3, outermost_linear = True).to(device)
optimizer = optim.Adam(siren.parameters(), lr = 1e-4)
num_epochs = 500

# Training loop.
for epoch in range(num_epochs):
    output, coords = siren(X)
    loss = nn.MSELoss(output, y)

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss {loss.item()}')
        grad = diff.gradient(output, coords)
        laplacian = diff.laplace(output, coords)

        # Display reconstructed image, its gradient and laplacian.
        fig, axes = plt.subplots(1, 3, figsize = (18, 6))
        axes[0].imshow(output.to(device).view(256, 256).detach().numpy())
        axes[1].imshow(grad.to(device).norm(dim = -1).to(device).view(256, 256).detach().numpy())
        axes[2].imshow(laplacian.view(256, 256).detach().numpy())
        plt.show()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


#####################################################################################################
# EXPERIMENT 2 : SOLVING THE POISSON EQUATION
#####################################################################################################

cameraman = utils.ImageFitting(128)
dataloader = DataLoader(cameraman, batch_size = 1, pin_memory = True, num_workers = 0)
X, y = next(iter(dataloader))
X, y = X.to(device), y.to(device)

siren = models.Siren(in_features = 2, out_features = 1, hidden_features = 256,
                     hidden_layers = 3, outermost_linear = True).to(device)
optimizer = optim.Adam(siren.parameters(), lr = 1e-4)
num_epochs = 1000

X, y = next(iter(dataloader))
X = X.to(device)
y = {key: value.to(device) for key, value in y.items()}

# Training loop.
for epoch in range(num_epochs):
    output, coords = siren(X)
    loss = utils.gradients_mse(output, coords, y['grads'])

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss {loss.item()}')
        grad = diff.gradient(output, coords)
        laplacian = diff.laplace(output, coords)

        # Display reconstructed image, its gradient and laplacian.
        fig, axes = plt.subplots(1, 3, figsize = (18, 6))
        axes[0].imshow(output.to(device).view(256, 256).detach().numpy())
        axes[1].imshow(grad.to(device).norm(dim = -1).to(device).view(256, 256).detach().numpy())
        axes[2].imshow(laplacian.view(256, 256).detach().numpy())
        plt.show()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


#####################################################################################################
# EXPERIMENT 3 : SOLVING THE WAVE EQUATION
#####################################################################################################
