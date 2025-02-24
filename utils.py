import torch
import scipy.ndimage
from torch.utils.data import Dataset
from PIL import Image
import skimage
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import matplotlib.pyplot as plt
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
    Returns the coordinates and source boundary values for the wave equation.
    '''
    def __init__(self, sidelength, source_coords = [0., 0.]):
        super().__init__()
        torch.manual_seed(0)
        self.sidelength = sidelength
        self.mgrid = get_mgrid(self.sidelength).detach()
        self.N_src_samples = 1000
        self.sigma = 5e-4
        self.source_coords = torch.tensor(source_coords).view(-1, 3)
        self.counter = 0
        self.full_count = 100e3

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = self.source_coords[0, 0]  # Time to apply  initial conditions.

        r = 5e2 * self.sigma * torch.rand(self.N_src_samples, 1).sqrt()
        phi = 2 * np.pi * torch.rand(self.N_src_samples, 1)

        # Circular sampling.
        source_coords_x = r * torch.cos(phi) + self.source_coords[0, 1]
        source_coords_y = r * torch.sin(phi) + self.source_coords[0, 2]
        source_coords = torch.cat((source_coords_x, source_coords_y), dim=1)

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.sidelength ** 2, 2).uniform_(-1, 1)

        # slowly grow time values from start time
        # this currently assumes start_time = 0 and max time value is 0.75. 
        time = torch.zeros(self.sidelength ** 2, 1).uniform_(0, 0.4 * (self.counter / self.full_count))
        coords = torch.cat((time, coords), dim=1)

        # make sure we always have training samples at the initial condition
        coords[-self.N_src_samples:, 1:] = source_coords
        coords[-2 * self.N_src_samples:, 0] = start_time

        # set up source
        normalize = 50 * gaussian(torch.zeros(1, 2), mu=torch.zeros(1, 2), sigma=self.sigma, d=2)
        boundary_values = gaussian(coords[:, 1:], mu=self.source_coords[:, 1:], sigma=self.sigma, d=2)[:, None]
        boundary_values /= normalize

        # only enforce initial conditions around start_time
        boundary_values = torch.where((coords[:, 0, None] == start_time), boundary_values, torch.Tensor([0]))
        dirichlet_mask = (coords[:, 0, None] == start_time)

        boundary_values[boundary_values < 1e-5] = 0.

        self.counter += 1

        return {'coords': coords}, {'source_boundary_values': boundary_values,
                                    'dirichlet_mask': dirichlet_mask}