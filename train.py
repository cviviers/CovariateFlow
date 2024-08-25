import os
import math
import time
import numpy as np
import random

## Imports for plotting
import seaborn as sns
sns.reset_orig()

## Progress bar
from tqdm import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision import datasets


from nflib import utils
from nflib import flows
from nflib import model
from nflib.utils import transforms as custom_transform

from utils.loaders import XrayImageDataset, CustomLoader_noise, TinyImageNet
import seaborn as sns
import pandas as pd
import wandb

import argparse

def train_flow(flow, config, train_loader, val_loader, checkpoint_path=None, model_name=None):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    flow.to(device)

    best_loss = torch.inf
    best_bpd = torch.inf

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(flow.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr'], steps_per_epoch=len(train_loader), epochs=config['epochs'], pct_start=0.05)

    # Check for a pre-trained model
    pretrained_filename = os.path.join(checkpoint_path, model_name + ".ckpt")

    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        ckpt = torch.load(pretrained_filename, map_location=device)
        flow.load_state_dict(ckpt['state_dict'])
    else:
        # Training loop
        print("Start training", model_name)
        flow.train()
        
        for epoch in range(config['epochs']):  

            running_loss = torch.tensor(0.0, device=device, requires_grad=False)
            running_bpd = torch.tensor(0.0, device=device, requires_grad=False)
            running_nll = torch.tensor(0.0, device=device, requires_grad=False)
            running_gradient_norm = torch.tensor(0.0, device=device, requires_grad=False)

            with tqdm(train_loader, unit="batch") as tepoch:
                wandb.log({'learning_rate': optimizer.param_groups[0]['lr']})
                for batch in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    inputs = batch[0].to(device)
                    inputs.requires_grad_(True)
                    optimizer.zero_grad()
                    ll = flow._get_likelihood(inputs, return_ll=True)
                    nll = (-ll)
                    mean_nll = nll.mean()
                    bpd = nll* np.log2(np.exp(1)) / np.prod(inputs.shape[-3:])
                    mean_bpd = bpd.mean()
                    mean_bpd.backward(retain_graph=True)
                    gradient_norm = torch.flatten(inputs.grad[:,1, ...], start_dim=1).norm(dim=1, p=2).mean(dim=0)
                    loss = mean_bpd + config['alpha'] * gradient_norm
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(flow.parameters(), config['max_norm'], norm_type=2.0, error_if_nonfinite=False, foreach=None)
                    optimizer.step()
                    scheduler.step()

                    running_loss+=loss
                    running_bpd+=mean_bpd
                    running_nll+=mean_nll
                    running_gradient_norm+=gradient_norm

                running_loss /= len(train_loader)
                running_bpd /= len(train_loader)
                running_nll /= len(train_loader)
                running_gradient_norm /= len(train_loader)

                wandb.log({'train_bpd': running_bpd})
                wandb.log({'train_nll': running_nll})
                wandb.log({'train_gradient_norm': running_gradient_norm})
                wandb.log({'train_loss': running_loss})

            if epoch % config['eval_epochs'] == 0:
                flow.eval()

                running_val_loss = torch.tensor(0.0, device=device, requires_grad=False)
                running_val_bpd = torch.tensor(0.0, device=device, requires_grad=False)
                running_val_nll = torch.tensor(0.0, device=device, requires_grad=False)
                running_val_gradient_norm = torch.tensor(0.0, device=device, requires_grad=False)

                with tqdm(val_loader, unit="batch") as vepoch:
                    for val_batch in vepoch:
                        vepoch.set_description(f"VAL Epoch {epoch}")
                        inputs = val_batch[0].to(device)
                        inputs.requires_grad_(True)
                        ll = flow._get_likelihood(inputs, return_ll=True)
                        nll = (-ll)
                        mean_nll = nll.mean()
                        bpd = nll* np.log2(np.exp(1)) / np.prod(inputs.shape[-3:])
                        mean_bpd = bpd.mean()
                        optimizer.zero_grad()
                        # Backward pass
                        # mean_nll.backward()
                        mean_bpd.backward()

                        gradient_norm = torch.flatten(inputs.grad[:,1, ...], start_dim=1).norm(dim=1, p=2).mean()

                        loss = mean_bpd + config['alpha'] * gradient_norm

                        running_val_loss += loss.item()
                        running_val_bpd += mean_bpd.item()
                        running_val_nll += mean_nll.item()
                        running_val_gradient_norm += gradient_norm.item()
                        optimizer.zero_grad()

                running_val_loss /= len(val_loader)
                running_val_bpd /= len(val_loader)
                running_val_nll /= len(val_loader)
                running_val_gradient_norm /= len(val_loader)


                wandb.log({'val_bpd': running_val_bpd})
                wandb.log({'val_nll': running_val_nll})
                wandb.log({'val_gradient_norm': running_val_gradient_norm})
                wandb.log({'val_loss': running_val_loss})

                if running_val_loss < best_loss:
                    print(f"New best loss model found {running_val_loss}, saving...")
                    torch.save({'state_dict': flow.state_dict()}, os.path.join(checkpoint_path, model_name + "_best_loss.ckpt"))
                    best_loss = running_val_loss

                if running_val_bpd < best_bpd:
                    print(f"New best bpd model found {running_val_bpd}, saving...")
                    torch.save({'state_dict': flow.state_dict()}, os.path.join(checkpoint_path, model_name + "_best_bpd.ckpt"))
                    best_bpd = running_val_bpd

                if epoch % config['epoch_save_freq'] == 0 and epoch > 0:
                    print(f"Saving checkpoint at epoch {epoch}")
                    torch.save({'state_dict': flow.state_dict()}, os.path.join(checkpoint_path, model_name + f"_epoch_{epoch}.ckpt"))
                        
                flow.train()
        # Save the model
        torch.save({'state_dict': flow.state_dict()}, os.path.join(checkpoint_path, model_name + "_last.ckpt"))

    # Test the model
    flow.eval()
    print("Start final validation testing", model_name)

    start_time = time.time()
    running_val_loss = torch.tensor(0.0, device=device, requires_grad=False)
    running_val_bpd = torch.tensor(0.0, device=device, requires_grad=False)
    running_val_nll = torch.tensor(0.0, device=device, requires_grad=False)
    running_val_gradient_norm = torch.tensor(0.0, device=device, requires_grad=False)

    # Testing on test_loader
    with tqdm(val_loader, unit="batch") as tepoch:
        for test_batch in tepoch:
    
            # Compute test metrics
            inputs = test_batch[0].to(device)
            inputs.requires_grad = True
            ll = flow._get_likelihood(inputs, return_ll=True)
            nll = (-ll)
            mean_nll = nll.mean()
            bpd = nll* np.log2(np.exp(1)) / np.prod(inputs.shape[-3:])
            mean_bpd = bpd.mean()
            optimizer.zero_grad()
            mean_bpd.backward()
            gradient_norm = torch.abs(torch.flatten(inputs.grad[:,1, ...], start_dim=1)).norm(dim=1, p=2).mean()
            loss = mean_bpd + config['alpha'] * gradient_norm
            running_val_loss += loss.item()
            running_val_bpd += bpd.mean().item()
            running_val_nll += mean_nll.item()
            running_val_gradient_norm += gradient_norm.item()

    running_val_loss /= len(val_loader)
    running_val_bpd /= len(val_loader)
    running_val_nll /= len(val_loader)
    running_val_gradient_norm /= len(val_loader)


    wandb.log({'test_bpd': mean_bpd})
    wandb.log({'test_nll': mean_nll})
    wandb.log({'test_gradient_norm': gradient_norm})
    wandb.log({'test_loss': loss})


    duration = time.time() - start_time
    result = {"val_loss": running_val_loss, "val_bpd": running_val_bpd, "val_nll":running_val_nll,  "val_gradient_norm":running_val_gradient_norm, "time": duration / len(val_loader) }

    return flow, result

# entree point for script
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train a conditional normalizing flow on MNIST')
    parser.add_argument('--train', type=bool, default=True, help='Train/Test model')
    parser.add_argument('--batch_size', type=int, default=160, help='Batch size')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=2, help='Typicality regularization parameter')
    parser.add_argument('--max_norm', type=float, default=100.0, help='Max norm for gradient clipping')
    parser.add_argument('--cond_coupling_layers', type=int, default=8, help='Number of coupling layers')
    parser.add_argument('--model_type', type=str, default='cond', help='cond: conditional, uncond: unconditional, uncond_sdl: unconditional with sdl')
    parser.add_argument('--data_type', type=int, default=0, help='0: filtered, 1 unfiltred')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='[MNIST, CIFAR10, TinyImageNet]')
    parser.add_argument('--sigma', type=float, default=1, help='Gaussian filter sigma')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to data')
    parser.add_argument('--epoch_save_freq', type=int, default=10, help='Frequency of saving additional checkpoints')

    args = parser.parse_args()

    wandb_run = wandb.init(project='CovariateFlow')

    config = {}
    config['alpha'] = args.alpha
    config['batch_size'] = args.batch_size
    config['epochs'] = args.epochs
    config['lr'] = args.lr
    config['max_norm'] = 100.0
    config['cond_coupling_layers'] = 8
    config['sigma'] = args.sigma
    config['eval_epochs'] = 5
    config['model_type'] = args.model_type
    config['data_type'] = args.data_type
    config['dataset'] = args.dataset
    config['data_path'] = args.data_path
    config['epoch_save_freq'] = args.epoch_save_freq
    wandb.config.update(config)

    # Setting the seed
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(0)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print("Using device:", device)

    # Transformations
    if args.data_type == 0:
        covariate_transform=transforms.Compose([custom_transform.pil_img_to_numpy, 
                                                custom_transform.normalize_8bit, 
                                                custom_transform.GaussianFilter(args.sigma),
                                                custom_transform.AdjustHighImage(),
                                                custom_transform.toTensor,
                                                custom_transform.ScaleAndQauntizeHigh(bits=16),
                                                custom_transform.Permute() ] )
    elif args.data_type == 1:
        covariate_transform=transforms.Compose([custom_transform.pil_img_to_numpy,
                                                   custom_transform.normalize_8bit,
                                                   custom_transform.toTensor,
                                                   custom_transform.ScaleAndQauntize(bits=16),
                                                   custom_transform.Permute()] )


    if args.dataset == 'CIFAR10':
        # DATASET
        dataset = datasets.CIFAR10(root=args.data_path, download=True, transform = covariate_transform, train=True)
        train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.85), int(len(dataset)*0.15)])
        # save the validation set
        val_set_path = os.path.join(args.data_path, 'val_set')
        if not os.path.exists(val_set_path):
            os.makedirs(val_set_path)
            np.save(os.path.join(val_set_path, 'val_set.npy'), val_set.indices)

    if args.dataset == 'TinyImageNet':
        train_set = TinyImageNet(root=args.data_path, transform = covariate_transform, split = 'train')
        val_set = TinyImageNet(root=args.data_path, transform = covariate_transform, split = 'val')
        # save the validation set
        val_set_path = os.path.join(args.data_path, 'val_set')
        if not os.path.exists(val_set_path):
            os.makedirs(val_set_path)
            np.save(os.path.join(val_set_path, 'val_set.npy'), val_set.indices)

    if args.dataset == 'MNIST':
        dataset = datasets.MNIST(root=args.data_path, download=True, transform = covariate_transform, train=True)
        train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.85), int(len(dataset)*0.15)])
        # save the validation set
        val_set_path = os.path.join(args.data_path, 'val_set')
        if not os.path.exists(val_set_path):
            os.makedirs(val_set_path)
            np.save(os.path.join(val_set_path, 'val_set.npy'), val_set.indices)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, drop_last=True, pin_memory=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, drop_last=False, num_workers=8)

    dataset_image_size = train_set[0][0].shape
 
    # Example
    if args.model_type == 'cond':
        noise_flow = model.create_conditional_flow_cifar(device, dataset_image_size, config['cond_coupling_layers'], train_set,)
        print("Created conditional flow")
    elif args.model_type == 'uncond':
        noise_flow = model.create_unconditional_flow_cifar(device, dataset_image_size, config['cond_coupling_layers'], train_set)
        print("Created unconditional flow")
    elif args.model_type == 'uncond_sdl':
        noise_flow = model.create_unconditional_sdl_flow_cifar(device, dataset_image_size, config['cond_coupling_layers'], train_set)
        print("Created unconditional + sdl flow")

    wandb.watch(noise_flow, log="all")

    # create new folder with the name of the run
    run_name = wandb.run.name
    model_name = f"CovariateFlow_{args.dataset}"
    new_checkpoint_path = f'./checkpoints/{model_name}/{run_name}'

    noise_flow, result = train_flow(noise_flow, config, train_loader, val_loader, checkpoint_path=new_checkpoint_path, model_name=model_name)
    wandb_run.finish()