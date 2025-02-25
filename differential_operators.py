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
    Returns jacobian of y w.r.t. x, of shape (batch_size, 1, 3).

    - y : torch.Tensor of shape (batch_size, 1) ;
    - x : torch.Tensor of shape (batch_size, 3) : the last dimension contains 2 space
    components and 1 time component.
    '''
    # Create empty jacobian tensor of shape (batch_size, 1, 3)
    batch_size = y.shape[0]
    jac = torch.zeros(batch_size, y.shape[-1], x.shape[-1], device = y.device)
    
    # calculate dy/dx over batches for each component of y.
    for i in range(y.shape[-1]):
        grad_outputs = torch.ones_like(y, device = x.device)
        jac[:, i, :] = torch.autograd.grad(y, x,
                                           grad_outputs = grad_outputs,
                                           create_graph = True,
                                           retain_graph = True)[0]

    return jac