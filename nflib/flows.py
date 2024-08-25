import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from nflib.utils import squeeze2d, create_quadrent_cycle_mask
import scipy
from . import thops


class CouplingLayer(nn.Module):

    def __init__(self, network, mask, c_in, output_mask_quadrant = None, is_conditioned=False):
        """
        Coupling layer inside a normalizing flow.
        Inputs:
            network - A PyTorch nn.Module constituting the deep neural network for mu and sigma.
                      Output shape should be twice the channel size as the input.
            mask - Binary mask (0 or 1) where 0 denotes that the element should be transformed,
                   while 1 means the latent will be used as input to the NN.
            c_in - Number of input channels
        """
        super().__init__()
        self.network = network
        self.scaling_factor = nn.Parameter(torch.zeros(c_in))
        # Register mask as buffer as it is a tensor which is not a parameter,
        # but should be part of the modules state.
        self.register_buffer('mask', mask)

        # if output_mask_quadrant is not None:
            
        self.register_buffer('output_mask_quadrant', output_mask_quadrant)

        self.is_conditioned = is_conditioned


    def forward(self, z, ldj, reverse=False, orig_img=None):
        """
        Inputs:
            z - Latent input to the flow
            ldj - The current ldj of the previous flows.
                  The ldj of this layer will be added to this tensor.
            reverse - If True, we apply the inverse of the layer.
            orig_img (optional) - Only needed in VarDeq. Allows external
                                  input to condition the flow on (e.g. original image)
        """
        # Apply network to masked input
        # print(z.shape, self.mask.shape)
        z_in = z * self.mask
        if orig_img is None:
            nn_out = self.network(z_in)
        else:
            nn_out = self.network(torch.cat([z_in, orig_img], dim=1))
        s, t = nn_out.chunk(2, dim=1)

        # Stabilize scaling output
        s_fac = self.scaling_factor.exp().view(1, -1, 1, 1)
        s = torch.tanh(s / s_fac) * s_fac

        if self.output_mask_quadrant is not None:
            output_qaudrant =  create_quadrent_cycle_mask(z.shape[2], z.shape[3], self.output_mask_quadrant)
            output_qaudrant = output_qaudrant.to(z.device)
            s = s * output_qaudrant
            t = t * output_qaudrant
        else:
            # Mask outputs (only transform the second part)
            s = s * (1 - self.mask)
            t = t * (1 - self.mask)

        # Affine transformation
        if not reverse:
            # Whether we first shift and then scale, or the other way round,
            # is a design choice, and usually does not have a big impact
            z = (z + t) * torch.exp(s)
            ldj += s.sum(dim=[1,2,3])
        else:
            z = (z * torch.exp(-s)) - t
            ldj -= s.sum(dim=[1,2,3])

        return z, ldj
    def is_conditioned(self):
        return self.is_conditioned  



class ConditionalAffineLayer(nn.Module):

    def __init__(self, network, c_in):
        """
        Coupling layer inside a normalizing flow.
        Inputs:
            network - A PyTorch nn.Module constituting the deep neural network for mu and sigma.
                      Output shape should be twice the channel size as the input.
            c_in - Number of input channels
        """
        super().__init__()
        self.network = network
        self.scaling_factor = nn.Parameter(torch.zeros(c_in))


    def forward(self, z, ldj, reverse=False, orig_img=None):
        """
        Inputs:
            z - Latent input to the flow
            ldj - The current ldj of the previous flows.
                  The ldj of this layer will be added to this tensor.
            reverse - If True, we apply the inverse of the layer.
            orig_img (optional) - Only needed in VarDeq. Allows external
                                  input to condition the flow on (e.g. original image)
        """
        
   
        nn_out = self.network(torch.cat([orig_img], dim=1))
        s, t = nn_out.chunk(2, dim=1)

        # Stabilize scaling output
        s_fac = self.scaling_factor.exp().view(1, -1, 1, 1)
        s = torch.tanh(s / s_fac) * s_fac

        # Affine transformation
        if not reverse:
            # Whether we first shift and then scale, or the other way round,
            # is a design choice, and usually does not have a big impact
            z = (z + t) * torch.exp(s)
            ldj += s.sum(dim=[1,2,3])
        else:
            z = (z * torch.exp(-s)) - t
            ldj -= s.sum(dim=[1,2,3])

        return z, ldj


class Dequantization(nn.Module):

    def __init__(self, alpha=1e-5, quants=256):
        """
        Inputs:
            alpha - small constant that is used to scale the original input.
                    Prevents dealing with values very close to 0 and 1 when inverting the sigmoid
            quants - Number of possible discrete values (usually 256 for 8-bit image)
        """
        super().__init__()
        self.alpha = alpha
        self.quants = quants

    def forward(self, z, ldj, reverse=False):
  
        # print(torch.amax(z), torch.amin(z))
        if not reverse:
            z, ldj = self.dequant(z, ldj)
            z, ldj = self.sigmoid(z, ldj, reverse=True)
        else:
            z, ldj = self.sigmoid(z, ldj, reverse=False)
            z = z * self.quants
            ldj += np.log(self.quants) * np.prod(z.shape[1:])
            z = torch.floor(z).clamp(min=-self.quants/2, max=self.quants/2-1).to(torch.float32)
        return z, ldj

    def sigmoid(self, z, ldj, reverse=False):
        # Applies an invertible sigmoid transformation
        if not reverse:
            ldj += (-z-2*F.softplus(-z)).sum(dim=[1,2,3])
            z = torch.sigmoid(z)
        else:
            z = z * (1 - self.alpha) + 0.5 * self.alpha  # Scale to prevent boundaries 0 and 1
            ldj += np.log(1 - self.alpha) * np.prod(z.shape[1:])
            ldj += (-torch.log(z) - torch.log(1-z)).sum(dim=[1,2,3])
            z = torch.log(z) - torch.log(1-z)
        return z, ldj

    def dequant(self, z, ldj):
        # Transform discrete values to continuous volumes
        z = z.to(torch.float32)
        z = z + torch.rand_like(z).detach()
        z = z / self.quants
        ldj -= np.log(self.quants) * np.prod(z.shape[1:])
        return z, ldj


class CustomDequantization(nn.Module):

    def __init__(self, alpha=1e-5, quants=np.power(2, 16)):
        """
        Inputs:
            alpha - small constant that is used to scale the original input.
                    Prevents dealing with values very close to 0 and 1 when inverting the sigmoid
            quants - Number of possible discrete values (usually 256 for 8-bit image)
        """
        super().__init__()
        
        self.alpha = alpha
        self.quants = quants

    def forward(self, z, ldj, reverse=False):

        if not reverse:
            # z, ldj = self.dequant(z, ldj)
            z, ldj = self.sigmoid(z, ldj, reverse=True)
        else:
            z, ldj = self.sigmoid(z, ldj, reverse=False)
            # z = z * self.quants
            # ldj += np.log(self.quants) * np.prod(z.shape[1:])
            # z = torch.floor(z).clamp(min=-self.quants/2, max=self.quants/2-1).to(torch.float32)
        return z, ldj

    def sigmoid(self, z, ldj, reverse=False):
        # Applies an invertible sigmoid transformation
        if not reverse:
            ldj += (-z-2*F.softplus(-z)).sum(dim=[1,2,3])
            z = torch.sigmoid(z)
        else:
            z = z * (1 - self.alpha) + 0.5 * self.alpha  # Scale to prevent boundaries 0 and 1
            ldj += np.log(1 - self.alpha) * np.prod(z.shape[1:])
            ldj += (-torch.log(z) - torch.log(1-z)).sum(dim=[1,2,3])
            z = torch.log(z) - torch.log(1-z)
        return z, ldj

    def dequant(self, z, ldj):
        # Transform discrete values to continuous volumes
        z = z.to(torch.float32)
        # z = z + torch.rand_like(z).detach()/self.quants
        # z = z / self.quants
        ldj -= np.log(self.quants) * np.prod(z.shape[1:])
        return z, ldj
    

class CustomDequantizationWithNoise(nn.Module):

    def __init__(self, alpha=1e-5, quants=np.power(2, 16)):
        """
        Inputs:
            alpha - small constant that is used to scale the original input.
                    Prevents dealing with values very close to 0 and 1 when inverting the sigmoid
            quants - Number of possible discrete values (usually 256 for 8-bit image)
        """
        super().__init__()
        self.alpha = alpha
        self.quants = quants

    def forward(self, z, ldj, reverse=False):
        if not reverse:
            z, ldj = self.dequant(z, ldj)
            z, ldj = self.sigmoid(z, ldj, reverse=True)
        else:
            z, ldj = self.sigmoid(z, ldj, reverse=False)
            # z = z * self.quants/quants
            ldj += np.log(self.quants) * np.prod(z.shape[1:])
            # z = torch.floor(z).clamp(min=-self.quants/2, max=self.quants/2-1).to(torch.float32)
        return z, ldj

    def sigmoid(self, z, ldj, reverse=False):
        # Applies an invertible sigmoid transformation
        if not reverse:
            ldj += (-z-2*F.softplus(-z)).sum(dim=[1,2,3])
            z = torch.sigmoid(z)
        else:
            z = z * (1 - self.alpha) + 0.5 * self.alpha  # Scale to prevent boundaries 0 and 1
            ldj += np.log(1 - self.alpha) * np.prod(z.shape[1:])
            ldj += (-torch.log(z) - torch.log(1-z)).sum(dim=[1,2,3])
            z = torch.log(z) - torch.log(1-z)
        return z, ldj

    def dequant(self, z, ldj):
        # Transform discrete values to continuous volumes
        z = z.to(torch.float32)
        z = z + torch.rand_like(z).detach()/self.quants
        # z = z / self.quants
        ldj -= np.log(self.quants) * np.prod(z.shape[1:])
        return z, ldj
    

class VariationalDequantization(Dequantization):

    def __init__(self, var_flows, alpha=1e-5, quants=256):
        """
        Inputs:
            var_flows - A list of flow transformations to use for modeling q(u|x)
            alpha - Small constant, see Dequantization for details
        """
        super().__init__(alpha=alpha)
        self.flows = nn.ModuleList(var_flows)
        self.quants = quants
    def dequant(self, z, ldj):
        z = z.to(torch.float32)
        img = (z / (self.quants-1)) * 2 - 1 # We condition the flows on x, i.e. the original image

        # Prior of u is a uniform distribution as before
        # As most flow transformations are defined on [-infinity,+infinity], we apply an inverse sigmoid first.
        deq_noise = torch.rand_like(z).detach()
        deq_noise, ldj = self.sigmoid(deq_noise, ldj, reverse=True)
        for flow in self.flows:
            deq_noise, ldj = flow(deq_noise, ldj, reverse=False, orig_img=img)
        deq_noise, ldj = self.sigmoid(deq_noise, ldj, reverse=False)

        # After the flows, apply u as in standard dequantization
        z = (z + deq_noise) / self.quants
        ldj -= np.log(self.quants) * np.prod(z.shape[1:])
        return z, ldj
    

class ConcatELU(nn.Module):
    """
    Activation function that applies ELU in both direction (inverted and plain).
    Allows non-linearity while providing strong gradients for any input (important for final convolution)
    """

    def forward(self, x):
        return torch.cat([F.elu(x), F.elu(-x)], dim=1)


class LayerNormChannels(nn.Module):

    def __init__(self, c_in):
        """
        This module applies layer norm across channels in an image. Has been shown to work well with ResNet connections.
        Inputs:
            c_in - Number of channels of the input
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(c_in)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class GatedConv(nn.Module):

    def __init__(self, c_in, c_hidden):
        """
        This module applies a two-layer convolutional ResNet block with input gate
        Inputs:
            c_in - Number of channels of the input
            c_hidden - Number of hidden dimensions we want to model (usually similar to c_in)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, kernel_size=3, padding=1),
            ConcatELU(),
            nn.Conv2d(2*c_hidden, 2*c_in, kernel_size=1)
        )

    def forward(self, x):
        out = self.net(x)
        val, gate = out.chunk(2, dim=1)
        return x + val * torch.sigmoid(gate)


class GatedConvNet(nn.Module):

    def __init__(self, c_in, c_hidden=32, c_out=-1, num_layers=3):
        """
        Module that summarizes the previous blocks to a full convolutional neural network.
        Inputs:
            c_in - Number of input channels
            c_hidden - Number of hidden dimensions to use within the network
            c_out - Number of output channels. If -1, 2 times the input channels are used (affine coupling)
            num_layers - Number of gated ResNet blocks to apply
        """
        super().__init__()
        c_out = c_out if c_out > 0 else 2 * c_in
        layers = []
        layers += [nn.Conv2d(c_in, c_hidden, kernel_size=3, padding=1)]
        for layer_index in range(num_layers):
            layers += [GatedConv(c_hidden, c_hidden),
                       LayerNormChannels(c_hidden)]
        layers += [ConcatELU(),
                   nn.Conv2d(2*c_hidden, c_out, kernel_size=3, padding=1)]
        self.nn = nn.Sequential(*layers)

        self.nn[-1].weight.data.zero_()
        self.nn[-1].bias.data.zero_()

    def forward(self, x):
        return self.nn(x)
    
class LinearNet(nn.Module):
    def __init__(self, c_in, c_hidden=32, c_out=-1, num_layers=1):
        super(LinearNet, self).__init__()
        layers = []

        layers+=[nn.Linear(c_in, c_hidden)]
        layers+=[nn.ReLU()]
        for layer_index in range(num_layers):
            layers+=[nn.Linear(c_hidden, c_hidden),
                       nn.ReLU()]
        layers+=[nn.Linear(c_hidden, c_out)]
        self.nn = nn.Sequential(*layers)
    def forward(self, x):
        return self.nn(x)
    
    
class SqueezeFlow(nn.Module):

    def forward(self, z, ldj, reverse=False):
        B, C, H, W = z.shape
        if not reverse:
            # Forward direction: H x W x C => H/2 x W/2 x 4C
            z = z.reshape(B, C, H//2, 2, W//2, 2)
            z = z.permute(0, 1, 3, 5, 2, 4)
            z = z.reshape(B, 4*C, H//2, W//2)
        else:
            # Reverse direction: H/2 x W/2 x 4C => H x W x C
            z = z.reshape(B, C//4, 2, 2, H, W)
            z = z.permute(0, 1, 4, 2, 5, 3)
            z = z.reshape(B, C//4, H*2, W*2)
        return z, ldj

class UnsqueezeFlow(nn.Module):

    def forward(self, z, ldj, reverse=False):
        B, C, H, W = z.shape
        if not reverse:
            # Forward direction: H/2 x W/2 x 4C => H x W x C
            z = z.reshape(B, C//4, 2, 2, H, W)
            z = z.permute(0, 1, 4, 2, 5, 3)
            z = z.reshape(B, C//4, H*2, W*2)
        else:
            # Reverse direction: H x W x C => H/2 x W/2 x 4C
            z = z.reshape(B, C, H//2, 2, W//2, 2)
            z = z.permute(0, 1, 3, 5, 2, 4)
            z = z.reshape(B, 4*C, H//2, W//2)
        return z, ldj

class SplitFlow(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

    def forward(self, z, ldj, reverse=False):
        if not reverse:
            z, z_split = z.chunk(2, dim=1)
            ldj += self.prior.log_prob(z_split).sum(dim=[1,2,3])
        else:
            z_split = self.prior.sample(sample_shape=z.shape).to(self.device)
            z = torch.cat([z, z_split], dim=1)
            ldj -= self.prior.log_prob(z_split).sum(dim=[1,2,3])
        return z, ldj

class CombineSplitFlow(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
        # self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

    def forward(self, z, ldj, reverse=False):
        if not reverse:
            z, z_split = z.chunk(2, dim=1)
            ldj += self.prior.log_prob(z_split).sum(dim=[1,2,3])
        else:
            z_split = self.prior.sample(sample_shape=z.shape).to(self.device)
            z = torch.cat([z, z_split], dim=1)
            ldj -= self.prior.log_prob(z_split).sum(dim=[1,2,3])
        return z, ldj



class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        # use the Q matrix from QR decomposition as the initial weight to make sure it's invertible
        # q, _ = torch.qr(weight)
        q, _ = torch.linalg.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, input, logdet, reverse=False):
        
        _,_, height, width = input.shape

        # You can also use torch.slogdet(self.weight)[1] to summarize the operations below\n",
        dlogdet = (
            height * width * torch.log(torch.abs(torch.det(self.weight.squeeze())))
        )

        if not reverse:
            out = F.conv2d(input, self.weight)
            logdet = logdet + dlogdet

        else:
            out = F.conv2d(input, self.weight.squeeze().inverse().unsqueeze(1).unsqueeze(2))
            logdet = logdet - dlogdet

        return out, logdet
    
class ActNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):

        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.log_scale.data.copy_(-std.clamp_(min=1e-6).log())

    def forward(self, input, logdet, reverse=False):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        dlogdet = height * width * torch.sum(self.log_scale)

        if not reverse:
            logdet += dlogdet
            return self.log_scale.exp() * (input + self.loc), logdet

        else:
            dlogdet *= -1
            logdet += dlogdet
            return input / self.log_scale.exp() - self.loc, logdet
        
class AffineCouplingSdl(nn.Module):

    def __init__( self):
        super(AffineCouplingSdl, self).__init__()

        self.b1 = nn.Parameter(torch.tensor(-5, dtype=torch.float32), requires_grad=True)
        self.b2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        #self.scale =   nn.Parameter(torch.tensor(1e-4, dtype=torch.float32) )

    def forward(self, x, ldj, imgs, reverse = False):

        #forward normalizing direction
        if not reverse:
            if imgs.shape[1] == 2 * x.shape[1]:  # needs squeezing
                imgs = squeeze2d(imgs, 2)

            scale = self.set_sdl_params(imgs)
            z = x
            if scale is not None:
                z *= scale
 
            if scale is None:
                log_abs_det_J = torch.tensor(0., dtype=torch.float32)
            else:
                log_abs_det_J = torch.sum(torch.log(scale), dim = (1, 2, 3))

            ldj += log_abs_det_J

        # inverse generative direction
        else:

            scale = self.set_sdl_params(imgs)
            if scale is not None:
                x /= scale
            z = x

            if scale is None:
                log_abs_det_J = torch.tensor(0., dtype=torch.float32)
            else:
                log_abs_det_J = torch.sum(torch.log(scale), dim = (1, 2, 3))
                
            ldj -= log_abs_det_J

        return z, ldj
    
    def set_sdl_params(self, imgs):
        """Set the initial values for beta params"""

        b1 = torch.sigmoid(self.b1) # Paper says to use exp, code uses sigmoid
        b2 = torch.sigmoid(self.b2)  # Paper says to use exp, code uses sigmoid

        scale = torch.sqrt(b1 * imgs + b2)

        return scale
    

class AffineCouplingGain(nn.Module):

    def __init__( self,):
        super(AffineCouplingGain, self).__init__()

        c = 1e-5
        self.g1 = nn.Parameter(torch.tensor(-5/c, dtype=torch.float32))
        self.g2 = nn.Parameter(torch.tensor(0/c, dtype=torch.float32))

    def forward(self, x, ldj, iso=None, reverse = False):

        #forward generative direction
        if not reverse:
            scale = self.set_gl_params(iso)
            shift = 0.0

            z = x
            if scale is not None:
                z *= scale
            if shift is not None:
                z += shift

            if scale is None:
                log_abs_det_J = torch.tensor(0., dtype=torch.float32)
            else:
                log_abs_det_J = torch.sum(torch.log(scale), dim = (1, 2, 3))

            ldj += log_abs_det_J

        # inverse
        else:

            scale = self.set_gl_params(iso)
            if scale is not None:
                x /= scale
            z = x

            if scale is None:
                log_abs_det_J = torch.tensor(0., dtype=torch.float32)
            else:
                log_abs_det_J = torch.sum(torch.log(scale), dim = (1, 2, 3))
                
            ldj -= log_abs_det_J


        return z, ldj
    
 
    
    def _set_gl_params(self, param):

        c = 1e-5
        # c = 1
        # c = 1e-2

        scale = torch.exp(c * self.g1) * param + torch.exp(c * self.g2)

        return scale


class ConditionalAffineCoupling(nn.Module):

    def __init__(self, network, mask, c_in, conditioning_network=None, condition_dim=None):
        """
        Coupling layer inside a normalizing flow.
        Inputs:
            network - A PyTorch nn.Module constituting the deep neural network for mu and sigma.
                      Output shape should be twice the channel size as the input.
            mask - Binary mask (0 or 1) where 0 denotes that the element should be transformed,
                   while 1 means the latent will be used as input to the NN.
            c_in - Number of input channels.
            condition - If provided, the coupling layer will be conditioned on the given condition.

        """
        super().__init__()
        self.network = network
        self.scaling_factor = nn.Parameter(torch.zeros(c_in))
        # Register mask as buffer as it is a tensor which is not a parameter,
        # but should be part of the modules state.
        self.register_buffer('mask', mask)
        if conditioning_network is not None:
            self.conditioning_network = conditioning_network
            self.conditioning_scaling_factor = nn.Parameter(torch.zeros(condition_dim))

    def forward(self, z, ldj, reverse=False, orig_img=None, condition=None):
        """
        Inputs:
            z - Latent input to the flow
            ldj - The current ldj of the previous flows.
                  The ldj of this layer will be added to this tensor.
            reverse - If True, we apply the inverse of the layer.
            orig_img (optional) - Only needed in VarDeq. Allows external
                                  input to condition the flow on (e.g. original image)
        """
        # Apply network to masked input
        # print(z.shape, self.mask.shape)
        z_in = z * self.mask

        if condition is not None:
            condition_output = self.conditioning_network(condition) # [B, ]
            condition_output = condition_output.reshape(condition_output.shape[0], z_in.shape[-2], z_in.shape[-1])
            if orig_img is None:
                nn_out = self.network(torch.cat([z_in, condition_output], dim=1))
            else:
                nn_out = self.network(torch.cat([z_in, orig_img, condition_output], dim=1))
        else:
            if orig_img is None:
                nn_out = self.network(z_in)
            else:
                nn_out = self.network(torch.cat([z_in, orig_img], dim=1))

        s, t = nn_out.chunk(2, dim=1)

        # Stabilize scaling output
        s_fac = self.scaling_factor.exp().view(1, -1, 1, 1)
        s = torch.tanh(s / s_fac) * s_fac

        # Mask outputs (only transform the second part)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)


        # Affine transformation
        if not reverse:
            # Whether we first shift and then scale, or the other way round,
            # is a design choice, and usually does not have a big impact
            z = (z + t) * torch.exp(s)
            ldj += s.sum(dim=[1,2,3])
        else:
            z = (z * torch.exp(-s)) - t
            ldj -= s.sum(dim=[1,2,3])

        return z, ldj
