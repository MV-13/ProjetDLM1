import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import scipy.ndimage
import skimage
import numpy as np
import matplotlib.pyplot as plt
import utils
import models
import differential_operators as diff
import loss_functions as loss_fn
import importlib

# Getting device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#####################################################################################################
# EXPERIMENT 1 : IMAGE FITTING 
#####################################################################################################

# Choose an image from skimage.data in the following list :
# camera, cat, astronaut, immunohistochemistry, brick, coffee, rocket.
img = utils.ImageFitting(256, skimage.data.cat())
dataloader = DataLoader(img, batch_size=1, pin_memory=True, num_workers=0)
X, y = next(iter(dataloader))
X, y = X.to(device), y.to(device)

# Instantiate siren model, ReLU model, optimizer and number of epochs.
siren = models.Siren(
    in_features=2, out_features=1,
    hidden_features=256, hidden_layers=3,
    outermost_linear=True).to(device)

standard = models.standard_network(
    activation=nn.ReLU(),
    in_features=2, out_features=1,
    hidden_features=256, hidden_layers=3,
    outermost_linear=True).to(device)

optim_siren = optim.Adam(siren.parameters(), lr=1e-4)
optim_std = optim.Adam(standard.parameters(), lr=1e-4)
num_epochs = 500

# Training loop.
for epoch in range(num_epochs):
    output_siren, coords_siren = siren(X)
    output_std, coords_std = standard(X)

    loss_siren = loss_fn.MSE(output_siren, y)
    loss_std = loss_fn.MSE(output_std, y)

    # Display results every 10 epochs.
    utils.display_img(epoch, 10, output_siren, coords_siren, loss_siren, 256, device)
    utils.display_img(epoch, 10, output_std, coords_std, loss_siren, 256, device)

    optim_siren.zero_grad()
    optim_std.zero_grad()
    loss_siren.backward()
    loss_std.backward()
    optim_siren.step()
    optim_std.step()


#####################################################################################################
# EXPERIMENT 2 : SOLVING THE POISSON EQUATION
#####################################################################################################

# Choose your image is previously.
img = utils.PoissonEqn(128, skimage.data.cat())
dataloader = DataLoader(img, batch_size=1, pin_memory=True, num_workers=0)
X, y = next(iter(dataloader))
X = X.to(device)
y = {key: value.to(device) for key, value in y.items()}

# Instantiate model, optimizer and number of epochs.
siren = models.Siren(in_features=2, out_features=1,
                     hidden_features=256, hidden_layers=3,
                     outermost_linear=True).to(device)

standard = models.standard_network(
    activation=nn.ReLU(),
    in_features=2, out_features=1,
    hidden_features=256, hidden_layers=3,
    outermost_linear=True).to(device)

optim_siren = optim.Adam(siren.parameters(), lr=1e-4)
optim_std = optim.Adam(standard.parameters(), lr=1e-4)
num_epochs = 1000

# Training loop.
for epoch in range(num_epochs):
    output_siren, coords_siren = siren(X)
    output_std, coords_std = standard(X)

    # The networks are supervised using the gradient of the image.
    loss_siren = loss_fn.gradients_mse(output_siren, coords_siren, y['grads'])
    loss_std = loss_fn.gradients_mse(output_std, coords_std, y['grads'])

    # Display results every 10 epochs.
    utils.display_img(epoch, 10, output_siren, coords_siren, loss_siren, 128, device)
    utils.display_img(epoch, 10, output_std, coords_std, loss_siren, 256, device)

    optim_siren.zero_grad()
    optim_std.zero_grad()
    loss_siren.backward()
    loss_std.backward()
    optim_siren.step()
    optim_std.step()


#####################################################################################################
# EXPERIMENT 3 : SOLVING THE WAVE EQUATION
#####################################################################################################

# Prepare wavefield dataset.
batch_size = int(1e4)
lambda1 = batch_size/100
lambda2 = batch_size/10
wavefield = utils.WaveSource(.4, .01, int(1e4))
dataloader = DataLoader(wavefield, batch_size=batch_size, shuffle=False)

# Instantiate model, optimizer and number of epochs.
siren = models.Siren(in_features=3, out_features=1, hidden_features=512,
                     hidden_layers=5, outermost_linear=True).to(device)
optimizer = optim.Adam(siren.parameters(), lr=2e-5)
num_epochs = 1000

# Training loop.
for epoch in range(num_epochs):
    for X in dataloader:
        X = X.to(device)
        output, coords = siren(X, detach_coords=False)

        loss = loss_fn.wave_loss(X, output, lambda1, lambda2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}, Loss {loss.item()}')


#####################################################################################################
# EXPERIMENT 4 : IMPLICIT INPAINTING
#####################################################################################################

# Get coordinates from the cameraman image (X) and the pixel values (y).
img = utils.ImageFitting(256, skimage.data.cat())
dataloader = DataLoader(img, batch_size=1, pin_memory=True, num_workers=0)
X, y = next(iter(dataloader))
X, y = X.to(device), y.to(device)
y.requires_grad_(True)

# Create mask.
mask = utils.mask(.99, 256)

# Instantiate model, optimizer and number of epochs.
siren = models.Siren(
    in_features=2, out_features=1,
    hidden_features=256, hidden_layers=3,
    outermost_linear=True).to(device)

standard = models.standard_network(
    activation=nn.ReLU(),
    in_features=2, out_features=1,
    hidden_features=256, hidden_layers=3,
    outermost_linear=True).to(device)

optim_siren = optim.Adam(siren.parameters(), lr=1e-4)
optim_std = optim.Adam(standard.parameters(), lr=1e-4)
num_epochs = 1000

# Training loop.
for epoch in range(num_epochs):
    output_siren, coords_siren = siren(X)
    output_std, coords_std = standard(X)

    # The networks are supervised using only non-masked pixels.
    loss_siren = loss_fn.MSE(mask * output_siren, mask * y)
    loss_std = loss_fn.MSE(mask * output_std, mask * y)

    # Display results every 10 epochs.
    utils.display_img(epoch, 10, output_siren, coords_siren, loss_siren, 256, device)
    utils.display_img(epoch, 10, output_std, coords_std, loss_siren, 256, device)

    optim_siren.zero_grad()
    optim_std.zero_grad()
    loss_siren.backward()
    loss_std.backward()
    optim_siren.step()
    optim_std.step()