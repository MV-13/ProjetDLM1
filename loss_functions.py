import torch
import differential_operators as diff


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

    - model_output: torch.Tensor, gradient from the model ;
    - coords: torch.Tensor ;
    - gt_gradients: torch.Tensor, ground-truth gradients.
    '''
    # Compute gradients on the model
    gradients = diff.gradient(model_output, coords)

    # Compare them with the ground-truth
    gradients_loss = torch.mean((gradients - gt_gradients).pow(2).sum(-1))

    return gradients_loss


#####################################################################################################
#####################################################################################################
def wave_loss(x, y, t, point_source, lambda1, lambda2):
    '''
    Loss function for the wave equation with initial conditions.

    - x : torch.Tensor, input vector of shape (batch size, 3) ;
    - y : torch.Tensor, output of the model of shape (batch size, 1) ;
    - lambda1 : float, weight for the dirichlet constraints, recommended value : batch size/100 ;
    - lambda2 : float, weight for the neuman constraints, recommended value : batch size/10.
    '''
    grad = diff.jacobian(y, x) # Shape (batch size, 1, 3).
    hess = diff.jacobian(grad[..., 0, :], x) # Shape (batch size, 3, 3).
    laplacian = hess[..., 1, 1, None] + hess[..., 2, 2, None] # Sum of space derivates.
    grad_t2 = hess[..., 0, 0, None] # Second derivates w.r.t. time.
    loss = grad_t2 - laplacian # Wave equation constraint.


    # Add initial condition constraints.
    dirichlet_loss = y[x[:, 0] == 0.] - point_source
    neumann_loss = grad[..., 0]
    loss += dirichlet_loss + neumann_loss

    return loss