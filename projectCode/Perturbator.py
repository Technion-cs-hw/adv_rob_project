from abc import abstractmethod

import torch
import torchvision.transforms.v2 as v2

class Perturbation:
    """
    A class for applying various perturbations to input data to test model robustness.
    """
    def __init__(self, seed=None):
        """
        Initializes the Perturbation class.
        
        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    @abstractmethod
    def apply(self, data):
        """
        Applies a specific type of perturbation to the data.
        
        Args:
            data (np.ndarray): Input data to perturb.
        Returns:
            torch.Tensor: Perturbed data.
        """
        return data
      
class Rotation(Perturbation):
    def __init__(self,seed=None,angle=0.0):
        super().__init__(seed)
        self.angle = angle

    def apply(self, data):
        pass

class Noise(Perturbation):
    def __init__(self, seed=None,noise_type = "Normal", noise_level=0.1):
        super().__init__(seed)
        self.noise_level = noise_level
        self.noise_type = noise_type

    def apply(self, data):
        """
        Adds Gaussian noise to the input tensor.
        
        Args:
            data (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Perturbed tensor with added noise.
        """
        if self.noise_type == "Uniform":
            noise = torch.rand(size=data.shape, generator=self.rng, device=data.device) * self.noise_level - self.noise_level/2
        elif self.noise_type == "Laplace":
            m = torch.distributions.laplace.Laplace(0, self.noise_level, validate_args=None)
            noise = m.rsample(data.shape)
        else:
            noise = torch.normal(0, self.noise_level, size=data.size(), generator=self.rng, device=data.device)
        return torch.clamp(data + noise,min = 0.0,max = 1.0)

class Mirror(Perturbation):
    def __init__(self, seed=None, axis=[-1]):
        super().__init__(seed)
        self.axis = axis

    def apply(self, data):
        """
        Flips the input tensor along a specified axis.
        
        Args:
            data (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Mirrored tensor.
        """
        return torch.flip(data, dims=self.axis)

class Patch(Perturbation):
    def __init__(self, seed=None, patch_size=(5,5)):
        super().__init__(seed)
        self.patch_size = patch_size

    def apply(self, data):
        """
        Replaces random patches in the input tensor with zeros.
        
        Args:
            data (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Perturbed tensor with zero patches.
        """
        B, C, H, W = data.size()
        
        patch_mask = torch.ones_like(data)
        
        top_left_y = torch.randint(0, H - self.patch_size[0] + 1, (B,))
        top_left_x = torch.randint(0, W - self.patch_size[1] + 1, (B,))
        
        patch_mask[:, :, top_left_y[0]:top_left_y[0] + self.patch_size[0], top_left_x[0]:top_left_x[0] + self.patch_size[1]] = 0
        return data * patch_mask
"""    
class Grayscale(Perturbation):
    def apply(self, data):
        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=data.device)
        return torch.sum(data * weights.view(1, -1, 1, 1), dim=1, keepdim=True)
"""

class Blur(Perturbation):
    def __init__(self, seed=None, kernel_size=3):
        super().__init__(seed)
        self.kernel_size = kernel_size

    def apply(self, data):
        """
        Applies a Gaussian blur to the input tensor.
        
        Args:
            data (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Blurred tensor.
        """
        #data = data / 255
        padding = self.kernel_size // 2
        kernel = torch.ones((data.size(1), 1, self.kernel_size, self.kernel_size), device=data.device)
        kernel /= self.kernel_size ** 2
        return torch.nn.functional.conv2d(data, kernel, padding=padding, groups=data.size(1))

class Perspective(Perturbation):
    def __init__(self, distortion_scale=0.6, p=1.0, seed=None):
        super().__init__(seed)
        self.transform = v2.RandomPerspective(distortion_scale=distortion_scale, p=p)

    def apply(self, data):
        return self.transform(data)

class RandomRotation(Perturbation):
    def __init__(self, degrees=(0, 180), seed=None):
        super().__init__(seed)
        self.transform = v2.RandomRotation(degrees=degrees)

    def apply(self, data):
        return self.transform(data)

class Grayscale(Perturbation):
    def __init__(self, seed=None):
        super().__init__(seed)
        self.transform = v2.Grayscale()

    def apply(self, data):
        return self.transform(data)

class ColorJitter(Perturbation):
    def __init__(self, brightness=0.5, hue=0.3, seed=None):
        super().__init__(seed)
        self.transform = v2.ColorJitter(brightness=brightness, hue=hue)

    def apply(self, data):
        return self.transform(data)
