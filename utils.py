import torch
import scipy.ndimage
from torch.utils.data import Dataset
from PIL import Image
import skimage
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import matplotlib.pyplot as plt
import numpy as np
import differential_operators as diff


#####################################################################################################
#####################################################################################################
def get_mgrid(sidelen, dim = 2):
    '''
    Returns a flattened grid of (x,y,...) coordinates in a range of -1 to 1.

    - sidelen : int, giving the size of the grid ;
    - dim : int, giving the dimension of the grid.
    '''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps = sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim = -1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


#####################################################################################################
#####################################################################################################
def get_cameraman_tensor(sidelength):
    '''
    Returns a tensor of the cameraman image from skimage.
    The image is resized to the given sidelength and normalized.

    - sidelength: int, gives the size of the image.
    '''
    img = Image.fromarray(skimage.data.camera())        
    transform = Compose([Resize(sidelength),
                         ToTensor(),
                         Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))])
    img = transform(img)
    return img


#####################################################################################################
#####################################################################################################
class ImageFitting(Dataset):
    '''
    Returns the pixel values of the cameraman image and their coordinates.

    - sidelength: int, gives the size of the image.
    '''
    def __init__(self, sidelength):
        super().__init__()
        img = get_cameraman_tensor(sidelength)
        self.pixels = img.permute(1, 2, 0).view(-1, 1) # Flatten image to pixels.
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1 # Dataset treated as a single sample.

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError # Only index 0 is allowed.
        return self.coords, self.pixels


#####################################################################################################
#####################################################################################################
class PoissonEqn(Dataset):
    '''
    Returns th pixels, gradients and laplacian of the cameraman image as a 1D-tensor and 
    their coordinates as a 2D-tensor.

    - sidelength : int, gives the size of the image.
    '''
    def __init__(self, sidelength):
        super().__init__()
        img = get_cameraman_tensor(sidelength)
        
        # Compute gradient horizontal and vertical gradients using sobel operator.       
        grads_x = scipy.ndimage.sobel(img.numpy(), axis = 1).squeeze(0)[..., None] # Horizontal.
        grads_y = scipy.ndimage.sobel(img.numpy(), axis = 2).squeeze(0)[..., None] # Vertical
        grads_x, grads_y = torch.from_numpy(grads_x), torch.from_numpy(grads_y)
        self.grads = torch.stack((grads_x, grads_y), dim = -1).view(-1, 2)

        # Compute laplacian using laplace operator.
        self.laplace = scipy.ndimage.laplace(img.numpy()).squeeze(0)[..., None]
        self.laplace = torch.from_numpy(self.laplace)
        
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.coords, {'pixels':self.pixels, 'grads':self.grads, 'laplace':self.laplace}


#####################################################################################################
#####################################################################################################
def display_img(epoch, step_til_summary, output, coords, loss, size, device):
    '''
    Plots the reconstructed image, its gradient and laplacian during training.

    - epoch : int, current epoch in the training loop ;
    - step_til_summary : int, how many iterations before a plot ;
    - output : torch.Tensor, reconstructed image from siren ;
    - coords : torch.Tensor, coordinates of the pixels ;
    - loss : torch.Tensor, loss value ;
    - size : int, size of the image to be displayed ;
    - device : torch.device, device used for computations
    '''
    if epoch % step_til_summary == 0:
        print(f'Epoch {epoch}, Loss {loss.item()}')
        grad = diff.gradient(output, coords)
        laplacian = diff.laplace(output, coords)

        # Display reconstructed image, its gradient and laplacian.
        fig, axes = plt.subplots(1, 3, figsize = (18, 6))
        axes[0].imshow(output.to(device).view(size, size).detach().numpy())
        axes[1].imshow(grad.to(device).norm(dim = -1).to(device).view(size, size).detach().numpy())
        axes[2].imshow(laplacian.view(size, size).detach().numpy())
        plt.show()


#####################################################################################################
#####################################################################################################
class WaveSource(Dataset):
    '''
    Creates a 2D-tensor dataset of shape (nb of points, 3).
    The last dimension corresponds to 2 space components and 1 time component.
    Space components are uniformly sampled between -1 and 1.
    The time component is slowly increased over time.

    - max_time : float, maximum time value ;
    - time_interval : float, time step between each observation ;
    - num_obs : nb of observations to sample for each time step.
    '''
    def __init__(self, max_time, time_interval, num_obs):
        super().__init__()
        # List of time steps.
        self.time_steps = torch.arange(0, max_time, time_interval)
        
        # Sample points for each time step.
        samples = []
        for t in self.time_steps:
            nb_pts = num_obs
            if t == 0.:
                # At time 0, we sample more points with a gaussian distribution.
                nb_pts *= 100
                space = torch.normal(mean = 0, std = 1e-3, size = (nb_pts, 2))
            
            else:
                # Otherwise, we sample with uniform distribution.
                space = torch.rand((nb_pts, 2)).uniform_(-1, 1)
            
            time = torch.full((nb_pts, 1), t)
            samples.append(torch.cat((time, space), dim = 1))
        
        self.samples = torch.cat(samples, dim = 0).requires_grad_(True)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        return self.samples[idx]


#####################################################################################################
#####################################################################################################
def gaussian(x, sigma = 5e-4):
    '''
    Returns the value of the 2D-gaussian function (0, sigma) evaluated in x.

    - x : torch.Tensor of shape (batch_size, 3), the last dimension contains 2 space
    components and 1 time component ;
    - sigma : float, variance of the gaussian ;
    '''
    distance = x[:, 1]**2 + x[:, 2]**2 # Computes distance to origin with space components.
    gaussian = torch.exp(-distance/2/sigma**2)/2/np.pi/sigma**2

    return gaussian