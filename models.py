import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#################################################################################################################
#################################################################################################################
class SineLayer(nn.Module):
    '''
    Defines a siren layer with a linear layer followed by a sine activation.

    - If is_first=True, omega_0 is a frequency factor which simply multiplies the activations
    before the nonlinearity ;
    - If is_first=False, then the weights will be divided by omega_0 so as to keep
    the magnitude of activations constant.
    '''
    def __init__(self, in_features, out_features, bias = True, is_first = False, omega_0 = 30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias = bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    

#################################################################################################################
#################################################################################################################
class Siren(nn.Module):
    '''
    Creates a siren model.

    - in_features: int, number of input features ;
    - hidden_features: int, number of neurons in the hidden layers ;
    - hidden_layers: int, number of hidden layers ;
    - out_features: int, number of output features ;
    - outermost_linear: bool, whether the last layer is a linear layer or not ;
    - first_omega_0: float, value of omega 0 for the first hidden layer ;
    - hidden_omega_0: float, value of omega 0 for the subsequent hidden layers.
    '''
    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 outermost_linear = False, first_omega_0 = 30, hidden_omega_0 = 30.):
        super().__init__()
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first = True,
                                  omega_0 = first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(
                hidden_features, hidden_features, 
                is_first = False,
                omega_0 = hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            # Initialize the weights of the last layer.
            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0, 
                    np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        
        else:
            self.net.append(SineLayer(
                hidden_features, out_features, 
                is_first=False,
                omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords, detach_coords=True):
        if detach_coords:
            coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output, coords


#################################################################################################################
#################################################################################################################
class standard_network(nn.Module):
    '''
    Creates a neural network.

    - in_features : int, number of input features ;
    - hidden_features : int, number of neurons in the hidden layers ;
    - hidden_layers : int, number of hidden layers ;
    - out_features : int, number of output features ;
    - outermost_linear : bool, whether the last layer is a linear layer or not ;
    '''
    def __init__(self, in_features, hidden_features, activation, hidden_layers, out_features,
                 outermost_linear = False):
        super().__init__()
        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features))
        self.net.append(activation)

        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(activation)

        if outermost_linear:
            self.net.append(nn.Linear(hidden_features, out_features))
        
        else:
            self.net.append(nn.Linear(hidden_features, out_features))
            self.net.append(activation)
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords, detach_coords=True):
        if detach_coords:
            coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output, coords