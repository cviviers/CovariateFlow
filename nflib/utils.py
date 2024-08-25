import numpy as np
import torch
import torchvision
from utils import transforms
import torchvision.transforms.functional as TF
import cv2
import pytorch_lightning as pl

def create_checkerboard_mask(h, w, invert=False):
    x, y = torch.arange(h, dtype=torch.int32), torch.arange(w, dtype=torch.int32)
    xx, yy = torch.meshgrid(x, y)
    mask = torch.fmod(xx + yy, 2)
    mask = mask.to(torch.float32).view(1, 1, h, w)
    if invert:
        mask = 1 - mask
    return mask

def create_quadrent_cycle_mask(h, w, quadrant):
    x, y = torch.arange(h, dtype=torch.int32), torch.arange(w, dtype=torch.int32)
    xx, yy = torch.meshgrid(x, y)
    mask = torch.zeros(h, w, dtype=torch.float32).view(1, 1, h, w)

    split_h = h // 2
    split_w = w // 2

    if quadrant == 0:
        mask[0, 0, :split_h, :split_w] = 1.0
    elif quadrant == 1:
        mask[0, 0, :split_h, split_w:] = 1.0
    elif quadrant == 2:
        mask[0, 0, split_h:, split_w:] = 1.0
    elif quadrant == 3:
        mask[0, 0, split_h:, :split_w] = 1.0
    return mask


# class MaskQuadrant:
#     def __init__(self, input_quadrant, output_quadrant):
#         self.type = MaskType.Quadrant
#         self.input_quadrant = input_quadrant
#         self.output_quadrant = output_quadrant

#     @staticmethod
#     def _get_quadrant_mask(quadrant, c, h, w):
#         b = torch.zeros((1, c, h, w), dtype=torch.float)
#         split_h = h // 2
#         split_w = w // 2
#         if quadrant == 0:
#             b[:, :, :split_h, :split_w] = 1.
#         elif quadrant == 1:
#             b[:, :, :split_h, split_w:] = 1.
#         elif quadrant == 2:
#             b[:, :, split_h:, split_w:] = 1.
#         elif quadrant == 3:
#             b[:, :, split_h:, :split_w] = 1.
#         else:
#             raise ValueError("Incorrect mask quadrant")
#         return b

#     def mask(self, x):
#         # x.shape = (bs, c, h, w)
#         c, h, w = x.size(1), x.size(2), x.size(3)
#         self.b_in = self._get_quadrant_mask(self.input_quadrant, c, h, w).to(x.device)
#         self.b_out = self._get_quadrant_mask(self.output_quadrant, c, h, w).to(x.device)
#         x_id = x * self.b_in
#         x_change = x * (1 - self.b_in)
#         return x_id, x_change

#     def unmask(self, x_id, x_change):
#         return x_id * self.b_in + x_change * (1 - self.b_in)
    
#     def mask_st_output(self, s, t):
#         return s * self.b_out, t * self.b_out


def create_channel_mask(c_in, invert=False):
    mask = torch.cat([torch.ones(c_in//2, dtype=torch.float32),
                      torch.zeros(c_in-c_in//2, dtype=torch.float32)])
    mask = mask.view(1, c_in, 1, 1)
    if invert:
        mask = 1 - mask
    return mask

def create_solid_mask(h, w, invert=False):
    mask = torch.ones(h, w, dtype=torch.float32).view(1, 1, h, w)
    mask = mask.to(torch.float32).view(1, 1, h, w)
    if invert:
        mask = 1 - mask
    return mask

def squeeze2d(x, factor=2, squeeze_type='chessboard', x_shape=None):
    assert factor >= 1
    if factor == 1:
        return x
    if x_shape is None:
        shape = x.shape
    else:
        shape = x_shape

    height = int(shape[1])
    width = int(shape[2])

    n_channels = int(shape[3])

    assert height % factor == 0 and width % factor == 0

    if squeeze_type == 'chessboard':
        # chess board
        x = torch.reshape(x, [-1, height // factor, factor,
                           width // factor, factor, n_channels])
        x = torch.transpose(x, [0, 1, 3, 5, 2, 4])

    elif squeeze_type == 'patch':
        # local patch
        x = torch.reshape(x, [-1, factor, height // factor,
                           factor, width // factor, n_channels])
        x = torch.transpose(x, [0, 2, 4, 5, 1, 3])
    else:
        
        print('Unknown squeeze type, using chessboard')
        # chess board
        x = torch.reshape(x, [-1, height // factor, factor,
                           width // factor, factor, n_channels])
        x = torch.transpose(x, [0, 1, 3, 5, 2, 4])

    x = torch.reshape(x, [-1, height // factor, width //
                       factor, n_channels * factor * factor])
    return x

class GenerateCallback(pl.Callback):
    
    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs # Images to reconstruct during training
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self._transform = transforms.NormalizeInverse(0.5, 0.5)
        self._noise_transform = transforms.NormalizeInverse(0.0, 0.5)
    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            # print(input_imgs.shape)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module.sample(img_shape=[input_imgs.shape[0],1,128,128], imgs=  input_imgs)
                pl_module.train()
            print(reconst_imgs.shape)
            # Plot and add to tensorboard
            input_imgs[:,:,0,...] =  self._transform(input_imgs[:,:,0,...])*255
            input_imgs[:,:,1,...] =  self._noise_transform(input_imgs[:,:,1,...])*255
            reconst_imgs = self._noise_transform(reconst_imgs)*255
            imgs = torch.cat([input_imgs.squeeze(1), reconst_imgs], dim=1).flatten(0,1).unsqueeze(1)
            imgs = imgs.type(torch.uint8)
            grid = torchvision.utils.make_grid(imgs, nrow=3, range=(0,255))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)