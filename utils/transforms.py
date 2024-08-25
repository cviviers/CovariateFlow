import torch
import numpy as np
import cv2
from typing import Any, Sequence
import random
import torchvision.transforms.functional as TF
import torchvision as TV
from .separateLowHighOHWT import lhs_2d
from .separateLowHighDWT import lhs_Haar_2d
from scipy.ndimage import gaussian_filter
from .loaders import ImageData
import skimage

# Convert images from 0-1 to 0-255 (integers)
def discretize_255(sample):
    return (sample * 255).astype(int)

def custom_reshape(img):
    og_img = img[0]
    img = img[1]
    return og_img, img.reshape(1,1, img.shape[0],img.shape[1]).type(torch.float)


class Scale_Image_Intensity(object):
    def __init__(self, scale) -> None:
        super().__init__()
        self.scale = scale
    def __call__(self, img):

        # img = img * np.random.normal(loc=self.mu, scale=self.std, size=img.shape)  
        return img * self.scale

class Decompose_Image(object):
    def __init__(self, depth, padLen) -> None:
        super().__init__()
        # self.levels = levels
        self.decomposition = lhs_Haar_2d(depth, padLen)
    def __call__(self, img):
        L, H = self.decomposition( img[1])
        # img = img * np.random.normal(loc=self.mu, scale=self.std, size=img.shape)  
        return img[0], L, H

# Conver numpy array to tensor
def toTensor(sample):
    return torch.from_numpy(sample.copy())


def twoToTensor(sample):
    # if len(sample.shape) == 3:
    #     return torch.from_numpy(sample.copy()).unsqueeze(1)
    # else:
    return torch.from_numpy(sample[0].copy()), torch.from_numpy(sample[1].copy())

def pil_img_to_numpy(sample):
    # np.asarray(sample)
    # print(np.asarray(sample).shape)
    # print(np.amax((np.asarray(sample))))
    return np.array(sample).copy().astype(float)


def normalize_8bit(sample):
    return (sample / 255).astype(np.float64)

def normalize_16bit(sample):
    return (sample / np.power(2, 16)).astype(np.float64)

def dequantize_255_two(sample):
    return (sample[0] / 255).to(torch.float64), (sample[1] / 255).to(torch.float64)

def dequantize_255_numpy(sample):
    return sample / 255

def discretize_255_noise(sample):
    return (sample[0] * 255).to(torch.int32).clip(0, 255), (sample[1] * 255).to(torch.int32).clip(0, 255)

def scale_mnist(sample):
    return sample*0.7

def create_noise(sample):
    gaussian_noise = torch.from_numpy(np.random.normal(loc=1, scale=0.1, size=sample.shape))
    noisy_img = torch.mul(gaussian_noise, sample)
    heteroscedastic_noise = noisy_img - sample
    return sample.type(torch.FloatTensor), heteroscedastic_noise.type(torch.FloatTensor)

#class Grayscale(object):
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def drop_dims(img):
    return img[:, :, :, 0]


class Quantize(object):
    def __init__(self, bits = 8) -> None:
        super().__init__()
        self.bits = bits

    def __call__(self, img):
        if self.bits == 8:
            return (img * 255).to(torch.int32).clip(0, 255)
        elif self.bits == 16:
            return (img * 65535).to(torch.int32).clip(0, 65535)
        elif self.bits == 32:
            return (img * 4294967295).to(torch.int32).clip(0, 4294967295)
        
class AddGaussianNoise(object):
    def __init__(self, std, mu, type_of_noise='multiplicative', return_noisy_img_only = False) -> None:
        super().__init__()
        self.std = std
        self.mu = mu
        self.type = type_of_noise
        self.return_noisy_img_only = return_noisy_img_only

    def __call__(self, img):
        if self.type == 'multiplicative':
            noise = np.random.normal(loc=self.mu, scale=self.std, size=img.shape)
            noisy_img = img*noise
        elif self.type == 'additive':
            noise = np.random.normal(loc=self.mu, scale=self.std, size=img.shape)
            noisy_img = img+noise
        
        return [img, noisy_img.clip(0, 1), noise] if not self.return_noisy_img_only else noisy_img.clip(0, 1)
    

class AddPoissonNoise(object):
    def __init__(self, peak = 0.3, bit_depth = 8, return_noisy_img_only = True) -> None:
        super().__init__()
        self.peak = peak
        self.bit_depth = bit_depth
        self.return_noisy_img_only = return_noisy_img_only

    def __call__(self, img):
        if self.bit_depth == 8:
            noise = np.random.poisson(np.uint8(img*np.power(2, self.bit_depth )))/np.power(2, self.bit_depth)*self.peak
        elif self.bit_depth == 16:
            noise = np.random.poisson(np.uint16(img*np.power(2, self.bit_depth )))/np.power(2, self.bit_depth)*self.peak

        # noise = np.random.poisson(img)/self.peak
        noisy_img = img + noise

        return [img, noisy_img, noise] if not self.return_noisy_img_only else noisy_img
    
class GaussianFilter(object):
    def __init__(self, sigma, channel_axis=2) -> None:
        super().__init__()
        # self.kernel_size = kernel_size
        self.sigma = sigma
        self.channel_axis = channel_axis

    def __call__(self, img):
        low = skimage.filters.gaussian(img,  sigma=self.sigma, channel_axis=self.channel_axis)
        high = img - low
        return np.stack([low, high], axis=0) 
    
    

class Drop_dims(object):
    def __call__(self, img):
        return img[:, :, :, 0]

class Img_and_noise_to_tensor(object):
    def __init__(self, data_type = torch.float32) -> None:
        self.data_type = data_type

    def __call__(self, img):
        
        return torch.from_numpy(img).to(self.data_type)
    
    
class ScaleAndQauntize(object):
    def __init__(self, bits = 8) -> None:
        super().__init__()
        self.bits = bits

    def __call__(self, img):
        if self.bits == 8:
            return (torch.floor(img * 255)).clip(0, 255).to(torch.float)
        elif self.bits == 16:
            return (torch.floor(img * 65535)).clip(0, 65535).to(torch.float16)
        elif self.bits == 32:
            return (torch.floor(img * 4294967295)).clip(0, 4294967295).to(torch.float32)

class ScaleAndQauntizeHigh(object):
    def __init__(self, bits = 8) -> None:
        super().__init__()
        self.bits = bits

    def __call__(self, img):
        if self.bits == 8:
            return torch.stack([img[0], (torch.floor(img[1] * 255)).clip(0, 255)], axis=0).to(torch.float32)
        elif self.bits == 16:
            return torch.stack([img[0], (torch.floor(img[1] * 65535)).clip(0, 65535)], axis=0).to(torch.float32)
        elif self.bits == 32:
            return torch.stack([img[0], (torch.floor(img[1] * 4294967295)).clip(0, 4294967295)], axis=0).to(torch.float32)
        
class ScaleAndQauntize(object):
    def __init__(self, bits = 8) -> None:
        super().__init__()
        self.bits = bits

    def __call__(self, img):
        if self.bits == 8:
            return torch.stack([torch.zeros_like(img), (torch.floor(img * 255)).clip(0, 255)], axis=0).to(torch.float32)
        elif self.bits == 16:
            return torch.stack([torch.zeros_like(img), (torch.floor(img * 65535)).clip(0, 65535)], axis=0).to(torch.float32)
        elif self.bits == 32:
            return torch.stack([torch.zeros_like(img), (torch.floor(img * 4294967295)).clip(0, 4294967295)], axis=0).to(torch.float32)

class PermuteCifar10(object):
    def __init__(self, permutation=(0, 3, 1, 2)) -> None:
        super().__init__()
        self.permutation = permutation

    def __call__(self, img):
        return img.permute(self.permutation)
    
class Permute(object):
    def __init__(self, permutation=(0, 3, 1, 2)) -> None:
        super().__init__()
        self.permutation = permutation

    def __call__(self, img):
        return img.permute(self.permutation)
    
class AdjustHighImage(object):
    def __init__(self, bits = 8) -> None:
        super().__init__()
        self.bits = bits
    def __call__(self, imgs) -> Any:
        return np.stack([imgs[0], ((imgs[1]+1)/2).clip(0, 1)], axis=0)
    
class NormalizeLowImg(object):
    def __init__(self, bits = 8) -> None:
        super().__init__()
        self.bits = bits

    def __call__(self, img):
        if self.bits == 8:
            return torch.stack([img[0] / 255, img[1]], axis=0)
        elif self.bits == 16:
            return torch.stack([img[0] / 65535, img[1]], axis=0)
        elif self.bits == 32:
            return torch.stack([img[0] / 4294967295, img[1]], axis=0)
            
class DropDimensions(object):
    def __init__(self, dims = [0, 1]) -> None:
        super().__init__()
        self.dims = dims

    def __call__(self, data):
        return data[self.dims, ...]
    
class Normalize_Img_and_noise(object):
    def __init__(self) -> None:
        self._img_norm = TV.transforms.Normalize(0.5, 0.5)
        self._noise_norm = TV.transforms.Normalize(0.5, 0.5)
    def __call__(self, data):

        return np.stack([data[0], data[1], data[2], self._img_norm( data[3][None, :, :])[0], self._noise_norm(data[4][None, :, :])[0]], axis=0)

class ToTuple(object):
    def __call__(self, img):
        return (img[0], img[1])

# def img_and_noise_to_tensor(img):
#     return torch.from_numpy(img[0]).to(torch.int32).clip(0, 255), torch.from_numpy(img[1]).to(torch.int32).clip(0, 255)

def scale_noise(sample):
    return sample[0], torch.clip(sample[1]+0.5, 0, 1)

# def high_pass_filter(patch):
#     low_comp = cv2.blur(patch, (7,7))
#     return low_comp, patch-low_comp

class High_pass_filter(object):
    def __init__(self, filter_size: int) -> None:
        super().__init__()
        self.kernel = np.ones((filter_size, filter_size),np.float64)/(filter_size**2)

    def __call__(self, img):

     

        low = cv2.filter2D(img, -1, self.kernel)
        high = img - low
        # low = np.empty_like(img)
        # high = np.empty_like(img)

        # for idx in range(img.shape[0]):
        #     low_comp = cv2.filter2D(img[idx], -1, self.kernel)
        #     low[idx, :, :] = low_comp
        #     high[idx, :, :] = img[idx] - low_comp
        return np.stack([low, high], axis=0)

class low_pass_filter(object):
    def __init__(self, filter_size: int):
        super().__init__()
        self.kernel = np.ones((filter_size, filter_size),np.float32)/(filter_size**2)

    def __call__(self, img):
        for idx in range(img.shape[0]):
            img[idx, :, :] = cv2.filter2D(img[idx], -1, self.kernel)
        return img


class MyRotateTransformNumpy:
    def __init__(self, times: Sequence[int]):
        self.times = times

    def __call__(self, x):
        k = random.choice(self.times)
        return np.rot90(x, k, axes=(-2,-1))

class MyRotateTransformTorchvision:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)
class AdjustNoise():
    def __init__(self, set_mean = 0.5):
        self.set_mean = set_mean
    def __call__(self, img):
        return np.stack([img[0], np.clip(img[1]+self.set_mean, a_min = 0, a_max=1)], axis=0)

class NormalizeInverse(TV.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std) #  1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
    

def GetImagesOnly(data):
    return np.stack([data.low_img, data.high_img], axis=0) 


