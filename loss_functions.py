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
    # compute gradients on the model
    gradients = diff.gradient(model_output, coords)
    # compare them with the ground-truth
    gradients_loss = torch.mean((gradients - gt_gradients).pow(2).sum(-1))
    return gradients_loss


#####################################################################################################
#####################################################################################################
