import pytorch_lightning as pl

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from nflib.flows import CouplingLayer, GatedConvNet, Dequantization,\
     VariationalDequantization, SqueezeFlow, SplitFlow, InvConv2d, AffineCouplingSdl, \
        UnsqueezeFlow, ConditionalAffineCoupling, LinearNet, CustomDequantization,CustomDequantizationWithNoise
from nflib.utils import create_checkerboard_mask, create_channel_mask, create_quadrent_cycle_mask

class CovariateFlow(torch.nn.Module):

    def __init__(self, flows, import_samples=8, train_set=None):
        """
        Inputs:
            flows - A list of flows (each a nn.Module) that should be applied on the images.
            import_samples - Number of importance samples to use during testing (see explanation below). Can be changed at any time
        """
        super().__init__()

        self.device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
        self.automatic_optimization=False
        
        self.flows = nn.ModuleList(flows)
        self.import_samples = import_samples
        # Create prior distribution for final latent space
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0, validate_args=True)
        # Example input for visualizing the graph
        if train_set is not None:
            self.example_input_array = train_set[0][0].unsqueeze(dim=0)

    def forward(self, imgs):
        # The forward function is only used for visualizing the graph
        return self._get_likelihood(imgs)

    def encode(self, imgs):
        # Given a batch of images, return the latent representation z and ldj of the transformations
        z, ldj = imgs[:, 1, ...], torch.zeros(imgs.shape[0], device=self.device)
        for flow in self.flows:
    
            if type(flow).__name__ == 'CouplingLayer':
                if flow.is_conditioned == True:
                    z, ldj = flow(z, ldj, orig_img = imgs[:, 0, ...], reverse=False)
                else:
                    z, ldj = flow(z, ldj, reverse=False)
            elif type(flow).__name__ == 'AffineCouplingSdl':
                z, ldj = flow(z, ldj, imgs = imgs[:, 0, ...], reverse=False)
            else:
                z, ldj = flow(z, ldj, reverse=False)
        return z, ldj

    def _get_likelihood(self, imgs, return_ll=False):
        """
        Given a batch of images, return the likelihood of those.
        If return_ll is True, this function returns the log likelihood of the input.
        Otherwise, the ouptut metric is bits per dimension (scaled negative log likelihood)
        """
        z, ldj = self.encode(imgs)
        log_pz = self.prior.log_prob(z).sum(dim=[1,2,3])
        log_px = ldj + log_pz
        if not return_ll:
            nll = -log_px
            # Calculating bits per dimension
            bpd = nll * np.log2(np.exp(1)) / np.prod(imgs.shape[-3:])
            return bpd.mean() 
        else:
            return log_px

    @torch.no_grad()
    def sample(self, img_shape, low_imgs, z_init=None):
        """
        Sample a batch of images from the flow.
        """
        # Sample latent representation from prior
        if z_init is None:
            z = self.prior.sample(sample_shape=img_shape).to(self.device)
        else:
            z = z_init.to(self.device)

        # Transform z to x by inverting the flows
        ldj = torch.zeros(img_shape[0], device=self.device)
        for flow in reversed(self.flows):
            if type(flow).__name__ == 'CouplingLayer':
                if flow.is_conditioned == True:
                    z, ldj = flow(z, ldj, orig_img = low_imgs, reverse=True)
                else:
                    z, ldj = flow(z, ldj, reverse=True)
            elif type(flow).__name__ == 'AffineCouplingSdl':
                z, ldj = flow(z, ldj, imgs = low_imgs, reverse=True)
            else:
                z, ldj = flow(z, ldj, reverse=True)
        return z
    
    def test_ood_per_sample(self, imgs, num_samples=8):
        """
        Estimate the log likelihood of the input images multiple times to get a better estimate.
        """
        samples = []
        for _ in range(num_samples):
            img_ll = self._get_likelihood(imgs, return_ll=True)
            samples.append(img_ll)
        img_ll = torch.stack(samples, dim=-1)
        img_ll = torch.logsumexp(img_ll, dim=-1) - np.log(num_samples)

        return img_ll




def create_conditional_flow(device, img_shape=(2, 3, 32, 32), num_coupling_layers = 6,  train_set = None,):

    num_images = img_shape[0]
    img_channels = img_shape[1]
    img_height = img_shape[2]
    img_width = img_shape[3]


    flow_layers = []
    flow_layers += [Dequantization(quants=np.power(2, 16))]
    for j in range(num_coupling_layers):
        flow_layers += [InvConv2d(in_channel=img_channels)]
        for i in range(2):
            
            flow_layers += [CouplingLayer(network=GatedConvNet(c_in=2*img_channels, c_out=2*img_channels, c_hidden=32, num_layers=4), mask = create_checkerboard_mask(h=img_height, w=img_width, invert=(i%2==1)), c_in=img_channels, is_conditioned=True)]  

    flow_layers += [AffineCouplingSdl()]
    flow_model = CovariateFlow(flows=flow_layers, train_set = train_set).to(device)

    return flow_model



def create_unconditional_flow(device, img_shape=(2, 3, 32, 32), num_coupling_layers = 6,  train_set = None,):

    num_images = img_shape[0]
    img_channels = img_shape[1]
    img_height = img_shape[2]
    img_width = img_shape[3]


    flow_layers = []
    flow_layers += [Dequantization(quants=np.power(2, 16))]
    for j in range(num_coupling_layers):
        flow_layers += [InvConv2d(in_channel=img_channels)]
        for i in range(2):
            
            flow_layers += [CouplingLayer(network=GatedConvNet(c_in=img_channels, c_out=2*img_channels, c_hidden=32, num_layers=4), mask = create_checkerboard_mask(h=img_height, w=img_width, invert=(i%2==1)), c_in=img_channels, is_conditioned=False)]  

    # flow_layers += [AffineCouplingSdl()]
    flow_model = CovariateFlow(flows=flow_layers, train_set = train_set).to(device)

    return flow_model

def create_unconditional_sdl_flow(device, img_shape=(2, 3, 32, 32), num_coupling_layers = 6,  train_set = None,):

    num_images = img_shape[0]
    img_channels = img_shape[1]
    img_height = img_shape[2]
    img_width = img_shape[3]


    flow_layers = []
    flow_layers += [Dequantization(quants=np.power(2, 16))]
    for j in range(num_coupling_layers):
        flow_layers += [InvConv2d(in_channel=img_channels)]
        for i in range(2):
            
            flow_layers += [CouplingLayer(network=GatedConvNet(c_in=img_channels, c_out=2*img_channels, c_hidden=32, num_layers=4), mask = create_checkerboard_mask(h=img_height, w=img_width, invert=(i%2==1)), c_in=img_channels, is_conditioned=False)]  

    flow_layers += [AffineCouplingSdl()]
    flow_model = CovariateFlow(flows=flow_layers, train_set = train_set).to(device)

    return flow_model



def create_conditional_cycle_test_flow(device, train_set = None):

    flow_layers = []

    flow_layers += [CustomDequantizationWithNoise()]


    for i in range(4):
        flow_layers += [InvConv2d(in_channel=1)]
        flow_layers += [CouplingLayer(network=GatedConvNet(c_in=1, c_out=2, c_hidden=64, num_layers=3), mask = create_checkerboard_mask(h=28, w=28, invert=(i%2==1)), c_in=1)] 

    for i in range(4):
        flow_layers += [InvConv2d(in_channel=1)]
        flow_layers += [CouplingLayer(network=GatedConvNet(c_in=2, c_out=2, c_hidden=64, num_layers=3), mask = create_checkerboard_mask(h=28, w=28, invert=(i%2==1)), c_in=1, 
                                      is_conditioned=True)]  

    for i in range(4):
        flow_layers += [InvConv2d(in_channel=1)]
        output_mask_quadrant = torch.tensor(i + 1) if i < 3 else torch.tensor(0) 

        flow_layers += [CouplingLayer(network=GatedConvNet(c_in=2, c_out=2, c_hidden=32, num_layers=3), mask = create_quadrent_cycle_mask(h=28, w=28, quadrant = i), 
                        c_in=1, is_conditioned=True, output_mask_quadrant = output_mask_quadrant)]  

    for i in range(2):
        flow_layers += [InvConv2d(in_channel=1)]
        flow_layers += [CouplingLayer(network=GatedConvNet(c_in=1, c_out=2, c_hidden=64, num_layers=3), mask = create_checkerboard_mask(h=28, w=28, invert=(i%2==1)), c_in=1)]  

    flow_layers += [AffineCouplingSdl()]
    flow_model = CovariateFlow(flows=flow_layers, train_set = train_set).to(device)
    return flow_model

