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
def get_mgrid(resolution, dim=2):
    '''
    Returns a flattened grid of (x, y,...) coordinates in a range of -1 to 1.

    - resolution : int, giving the size of the grid ;
    - dim : int, giving the dimension of the grid.
    '''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=resolution)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


#####################################################################################################
#####################################################################################################
def get_image_tensor(resolution, imgchoice=skimage.data.camera() ):
    '''
    Returns a tensor of the image from skimage.
    The image is resized to the given sidelength and normalized.

    - resolution: int, gives the size of the image.
    '''
    # Convert image to grayscale if it has 3 channels.
    if len(imgchoice.shape) == 3 and imgchoice.shape[2] == 3:
        imgchoice = skimage.color.rgb2gray(imgchoice) 

    # Convert numpy array to PIL image.
    if isinstance(imgchoice, np.ndarray):
        imgchoice = Image.fromarray((imgchoice * 255).astype(np.uint8))

    # Resize image and normalize.
    imgchoice = imgchoice.resize((resolution, resolution), Image.BILINEAR)
    imgchoice = 1-(np.array(imgchoice)/255.0)
    img = Image.fromarray(imgchoice)
    transform = Compose([
        Resize(resolution),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))])
    img = transform(img)

    return img


#####################################################################################################
#####################################################################################################
class ImageFitting(Dataset):
    '''
    Returns the pixel values of the chosen image and their coordinates.

    - resolution: int, gives the size of the image.
    '''
    def __init__(self, resolution, imgchoice):
        super().__init__()
        img = get_image_tensor(resolution, imgchoice)
        self.pixels = img.permute(1, 2, 0).view(-1, 1) # Flatten image.
        self.coords = get_mgrid(resolution, 2)

    def __len__(self):
        return 1 # Dataset treated as a single sample.

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError # Only index 0 is allowed.
        return self.coords, self.pixels


#####################################################################################################
#####################################################################################################
class PoissonEqn(Dataset):
    '''
    Returns the pixels, gradients and laplacian of the chosen image as a dictionary
    of 1D-tensors and its coordinates as a 2D-tensor.

    - resolution: int, gives the size of the image.
    '''
    def __init__(self, resolution, imgchoice):
        super().__init__()
        img = get_image_tensor(resolution, imgchoice)
        
        # Compute horizontal and vertical gradients using sobel operator.       
        grads_x = scipy.ndimage.sobel(img.numpy(), axis=1).squeeze(0)[..., None] # Horizontal.
        grads_y = scipy.ndimage.sobel(img.numpy(), axis=2).squeeze(0)[..., None] # Vertical
        grads_x, grads_y = torch.from_numpy(grads_x), torch.from_numpy(grads_y)
        self.grads = torch.stack((grads_x, grads_y), dim=-1).view(-1, 2)

        # Compute laplacian using laplace operator.
        self.laplace = scipy.ndimage.laplace(img.numpy()).squeeze(0)[..., None]
        self.laplace = torch.from_numpy(self.laplace)
        
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(resolution, 2)

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
    - output : torch.Tensor, reconstructed image from the model ;
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
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(output.to(device).view(size, size).detach().numpy())
        axes[1].imshow(grad.to(device).norm(dim=-1).to(device).view(size, size).detach().numpy())
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

    - max_time: float, maximum time value ;
    - time_interval: float, time step between each observation ;
    - num_obs: nb of observations to sample for each time step.
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
                space = torch.rand((nb_pts, 2)).uniform_(-.01, .01)
            
            else:
                # Otherwise, we sample with uniform distribution.
                space = torch.rand((nb_pts, 2)).uniform_(-1, 1)
            
            time = torch.full((nb_pts, 1), t)
            samples.append(torch.cat((time, space), dim=1))
        
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

    - x: torch.Tensor of shape (batch_size, 3), the last dimension contains 2 space
    components and 1 time component ;
    - sigma: float, variance of the gaussian.
    '''
    distance = x[:, 1]**2 + x[:, 2]**2 # Computes distance to origin with space components.
    gaussian = torch.exp(-distance/2/sigma**2)/np.sqrt(2*np.pi)/sigma/50
    clipped_gaussian = torch.where(gaussian < 1e-5, torch.tensor(0.0), gaussian)

    return clipped_gaussian


#####################################################################################################
#####################################################################################################
def mask(ratio, img_size):
    '''
    Returns a mask of shape (1, img_size**2, 1) with random pixels uniformly set to 0.

    - ratio: float between 0.0 and 1.0, the proportion of pixels to mask ;
    - img_size: int, size of the image to which the mask will be applied ;
    '''
    num_masked_pixels = int(img_size**2 * ratio)
    mask = torch.ones(img_size**2, dtype=torch.float32)
    masked_idx = torch.randperm(img_size**2)[:num_masked_pixels]
    mask[masked_idx] = 0.0
    mask = mask.view(1, img_size**2, 1)
    mask.requires_grad_(True)

    return mask