import torch
import differential_operators as diff
import utils


#####################################################################################################
#####################################################################################################
def MSE(X, y):
    '''
    Returns the mean squared error between X and y.

    - X : torch.Tensor, output of a model ;
    - y : torch.Tensor, ground-truth.
    '''
    return ((X - y)**2).mean()


#####################################################################################################
#####################################################################################################
def gradients_mse(model_output, coords, gt_gradients):
    '''
    Returns the mean squared error between the gradients computed by the model and
    the ground-truth gradients.

    - model_output: torch.Tensor, output of a model ;
    - coords: torch.Tensor, input coordinates for the model ;
    - gt_gradients: torch.Tensor, ground-truth gradients.
    '''
    # Compute gradients on the model
    gradients = diff.gradient(model_output, coords)

    # Compare them with the ground-truth
    gradients_loss = torch.mean((gradients - gt_gradients).pow(2).sum(-1))

    return gradients_loss


#####################################################################################################
#####################################################################################################
def wave_loss(x, y, lambda1, lambda2):
    '''
    Loss function for the wave equation with initial conditions.

    - x: torch.Tensor, input vector of shape (batch size, 3) ;
    - y: torch.Tensor, output of the model of shape (batch size, 1) ;
    - lambda1: float, weight for the dirichlet constraints, recommended value: batch size/100 ;
    - lambda2: float, weight for the neuman constraints, recommended value: batch size/10.
    '''
    grad = diff.jacobian(y, x) # Shape (batch size, 1, 3).
    hess = diff.jacobian(grad[:, 0, :], x) # Shape (batch size, 3, 3).
    laplacian = hess[..., 1, 1, None] + hess[..., 2, 2, None] # Sum of space derivatives.
    grad_t2 = hess[..., 0, 0, None] # Second derivates w.r.t. time.
    eq_loss = torch.abs(grad_t2 - laplacian).sum() # Wave equation constraint.

    # Select instances where time = 0.
    t0_mask = (x[:, 0] == 0.).view(-1, 1)
    y0 = torch.where(t0_mask, y, torch.tensor(0.0, device=y.device))
    X0 = torch.where(t0_mask, x, torch.tensor(0.0, device=x.device))

    # Dirichlet and Neuman initial constraints.
    dirichlet_loss = lambda1 * torch.abs(y0 - utils.gaussian(X0).unsqueeze(1)).sum()
    neuman_loss = lambda2 * torch.abs(grad[..., 0] * t0_mask).sum() # Derivative w.r.t. time.
    
    loss = eq_loss + dirichlet_loss + neuman_loss

    return loss