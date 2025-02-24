import torch


#####################################################################################################
#####################################################################################################
def gradient(y, x, grad_outputs = None):
    '''
    Returns the gradient of y w.r.t. x.

    - y : torch.Tensor representing the vector field with shape (..., N) where N represents
    the components of the vector field ;
    - x : torch.Tensor representing the spatial coordinates ;
    - grad_outputs : if None, return a constant gradient with 1s.
    '''
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph = True)[0]
    return grad


#####################################################################################################
#####################################################################################################
def divergence(y, x):
    '''
    Returns the divergence of y w.r.t. x.
    The divergence corresponds to the sum of the derivatives of the components of y w.r.t.
    the components of x.

    - y : torch.Tensor representing the vector field with shape (..., N) where N represents
    the components of the vector field ;
    - x : torch.Tensor representing the spatial coordinates.
    '''
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


#####################################################################################################
#####################################################################################################
def laplace(y, x):
    '''
    Returns the Laplacian of y w.r.t. x.

    - y : torch.Tensor ;
    - x : torch.Tensor.
    '''
    grad = gradient(y, x)
    return divergence(grad, x)


#####################################################################################################
#####################################################################################################
def jacobian(y, x):
    '''
    Returns jacobian of y w.r.t. x.

    - y : torch.Tensor, last dimension must be number of components ;
    - x : torch.Tensor, last dimension must be number of components.
    '''
    # Create empty jacobian tensor of shape (batch, observations, y components, x components)
    batch_size, num_observations = y.shape[:2]
    jac = torch.zeros(batch_size, num_observations, y.shape[-1], x.shape[-1]).to(y.device)
    
    # calculate dy/dx over batches for each component of y.
    for i in range(y.shape[-1]):
        y_flat = y[...,i].view(-1, 1)
        jac[:, :, i, :] = grad(y_flat, x, torch.ones_like(y_flat), create_graph = True)[0]

    return jac