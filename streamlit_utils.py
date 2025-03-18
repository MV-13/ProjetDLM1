import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import skimage.data
import importlib

import torch.nn as nn
import scipy.ndimage
import skimage
import numpy as np
import matplotlib.pyplot as plt
import utils
import models
import differential_operators as diff
import loss_functions as loss_fn

###############################################################################################
###############################################################################################
def choose_image(image_option):
    if image_option == "camera":
        choice = skimage.data.camera()
    elif image_option == "cat":
        choice = skimage.data.cat()
    elif image_option == "astronaut":
        choice = skimage.data.astronaut()
    elif image_option == "immunohistochemistry":
        choice = skimage.data.immunohistochemistry()
    elif image_option == "brick":
        choice = skimage.data.brick()
    elif image_option == "coffee":
        choice = skimage.data.coffee()
    elif image_option == "rocket":
        choice = skimage.data.rocket()

    return choice


###############################################################################################
###############################################################################################
def choose_function(activation_option):
    if activation_option == "ReLU":
        choice = nn.ReLU()
    elif activation_option == "Sigmoid":
        choice = nn.Sigmoid()
    elif activation_option == "Tanh":
        choice = nn.Tanh()
    elif activation_option == "LeakyReLU":
        choice = nn.LeakyReLU()

    return choice