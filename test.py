import torch
from torchvision import transforms
from torchvision import datasets
import time
from tqdm import tqdm
import numpy as np
import argparse
import json
import os
from nflib.utils import transforms as custom_transform
from utils.loaders import CIFAR10C, TinyImageNet, TinyImageNetC
from nflib import model
from sklearn.metrics import roc_curve, average_precision_score, roc_auc_score

# evaluate model
def test_dataset(model, test_loader, device, val_stats = None, num_evals = 1, name = None, output_path = None):
    # Set device
    model.to(device)
    start_time = time.time()

    if val_stats is not None:
        val_mean_bpd = torch.tensor(val_stats['mean_bpd']).to(device)
        val_std_bpd = torch.tensor(val_stats['std_bpd']).to(device)
        val_mean_grad = torch.tensor(val_stats['mean_grad']).to(device)
        val_std_grad =  torch.tensor(val_stats['std_grad']).to(device)
        

    test_nll = []
    test_gradient_norm = []
    test_bpd = []
    test_nsd = []

    model.eval()
    
    with tqdm(test_loader, unit="batch") as tepoch:
        for test_batch in tepoch:
    
            inputs = test_batch[0].to(device)
            inputs.requires_grad = True
            ll = model.test_ood_per_sample(inputs, num_evals)
            # ll = model._get_likelihood(inputs, return_ll=True)
            nll = (-ll)
            mean_nll = nll.mean()
            test_nll.append(mean_nll.item())

            bpd = nll* np.log2(np.exp(1)) / np.prod(inputs.shape[-3:])
            mean_bpd = bpd.mean()
            test_bpd.append(mean_bpd.item())
            
            # Backward pass
            mean_bpd.backward()
            flattened_tensor = torch.flatten(inputs.grad[:,1, ...], start_dim=1)
            gradient_norm = torch.mean(flattened_tensor.norm(dim=1, p=2))
            test_gradient_norm.append(gradient_norm.item())

            # Compute NSD if val_stats is provided
            if val_stats is not None:
                normalized_bpd = (mean_bpd - val_mean_bpd) / val_std_bpd
                normalized_grad = (gradient_norm - val_mean_grad) / val_std_grad
                abs_normalized_new_bpd = torch.abs(normalized_bpd)
                abs_normalized_new_grad = torch.abs(normalized_grad)
                nsd = torch.add(abs_normalized_new_bpd, abs_normalized_new_grad)
                test_nsd.append(nsd.item())

    # write output as npy files
    if output_path is not None:
        nll_path = os.path.join(output_path, name + '_nll.npy')
        np.save(nll_path, np.array(test_nll))
        bpd_path = os.path.join(output_path, name + '_bpd.npy')
        np.save(bpd_path, np.array(test_bpd))
        grad_path = os.path.join(output_path, name + '_grad.npy')
        np.save(grad_path, np.array(test_gradient_norm))

        if val_stats is not None:
            nsd_path = os.path.join(output_path, name + '_nsd.npy')
            np.save(nsd_path, np.array(test_nsd))

    duration = time.time() - start_time
    result = {"time": duration / len(test_loader), 'test_lls': np.array(test_nll), 'test_bpd':np.array( test_bpd), 
              'mean_bpd': np.mean(test_bpd), 'std_bpd': np.std(test_bpd), 'test_grad': np.array(test_gradient_norm),  
              'mean_grad': np.mean(test_gradient_norm), 'std_grad': np.std(test_gradient_norm), 'test_nsd': np.array(test_nsd)}
    return result

def read_validation_stats(path):
    with open(path, "r") as f:
        val_stats_data =  json.load(f)
    return val_stats_data

def compute_scores(test_results):

    score_auroc = []
    score_fpr = []

    bpd_auroc = []
    bpd_fpr = []

    grad_auroc = []
    grad_fpr = []

    for key in test_results.keys():
        if 'score' in key:
            score_auroc.append(test_results[key]['auroc'])
            score_fpr.append(test_results[key]['fpr'])
        elif 'bpd' in key:
            bpd_auroc.append(test_results[key]['auroc'])
            bpd_fpr.append(test_results[key]['fpr'])
        elif 'gradient' in key:
            grad_auroc.append(test_results[key]['auroc'])
            grad_fpr.append(test_results[key]['fpr'])

    return np.mean(score_auroc), np.mean(score_fpr), np.mean(bpd_auroc), np.mean(bpd_fpr), np.mean(grad_auroc), np.mean(grad_fpr)
      

def compute_results(val_mean_bpd, val_std_bpd, val_mean_grad, val_std_grad, test_bpd, test_grad, corrupt_bpds,
                     corrupt_grads, corruption, severity, corruption_combined_results):

    bpd_data = np.nan_to_num(corrupt_bpds)
    grads_data = np.nan_to_num(corrupt_grads)

    y_true = np.zeros(len(test_bpd))
    y_true = np.append(y_true, np.ones(len(bpd_data)))
    y_score = np.append(test_bpd, bpd_data)
    auroc_bpd = roc_auc_score(y_true, y_score)
    print(f"{auroc_bpd=}")
    # calucalte AUPR (area under the precisionrecall curve)
    aupr_bpd = average_precision_score(y_true, y_score)
    # FPR at 95% TPR (True Negative Rateat a fixed level of 95% True Positive Rate).
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fpr_bpd =  fpr[np.argmax(tpr >= 0.95)]
    corruption_combined_results[f"{corruption}_{severity}_bpd"] = {"auroc": auroc_bpd, "aupr": aupr_bpd, "fpr": fpr_bpd}

    # calculate everything with test_gradient_norms
    y_score = np.append(test_grad, grads_data)
    auroc_grad = roc_auc_score(y_true, y_score)
    print(f"{auroc_grad=}")
    # calucalte AUPR (area under the precisionrecall curve)
    aupr = average_precision_score(y_true, y_score)
    # FPR at 95% TPR (True Negative Rateat a fixed level of 95% True Positive Rate).
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fpr_grad =  fpr[np.argmax(tpr >= 0.95)]
    abs_grad_distances = np.mean(np.abs(test_grad - grads_data))
    grad_distances = np.mean(test_grad - grads_data)
    corruption_combined_results[f"{corruption}_{severity}_gradient"] = {"auroc": auroc_grad, "aupr": aupr, "fpr": fpr_grad}

    normalized_new_bpd = (bpd_data - val_mean_bpd) / val_std_bpd
    normalized_test_bpd = (test_bpd - val_mean_bpd) / val_std_bpd
    normalized_new_grad = (grads_data - val_mean_grad) / val_std_grad
    normalized_test_grad = (test_grad - val_mean_grad) / val_std_grad
    abs_normalized_new_bpd = np.abs(normalized_new_bpd)
    abs_normalized_test_bpd = np.abs(normalized_test_bpd)
    abs_normalized_new_grad = np.abs(normalized_new_grad)
    abs_normalized_test_grad = np.abs(normalized_test_grad)
    summed_normalized = np.add(abs_normalized_new_bpd, abs_normalized_new_grad)
    summed_normalized_test = np.add(abs_normalized_test_bpd, abs_normalized_test_grad)

    # compute the roc auc score
    y_true = np.zeros(len(summed_normalized_test))
    y_true = np.append(y_true, np.ones(len(summed_normalized)))
    y_score = np.append(summed_normalized_test, summed_normalized)
    auroc_score = roc_auc_score(y_true, y_score)
    print(f"{auroc_score=}")
    # calucalte AUPR (area under the precisionrecall curve)
    aupr_score = average_precision_score(y_true, y_score)
    # FPR at 95% TPR (True Negative Rateat a fixed level of 95% True Positive Rate).
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fpr_score =  fpr[np.argmax(tpr >= 0.95)]
    corruption_combined_results[f"{corruption}_{severity}_score"] = {"auroc": auroc_score, "aupr": aupr_score, "fpr": fpr_score}

    return corruption_combined_results

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test CovariateFlow')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--validation_stats_path', type=str, default=None, required=False, help='Path to validation stats')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data')
    parser.add_argument('--dataset', type=str, required=True, default='CIFAR10', help='Dataset name (CIFAR10 or ImageNet200)')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output')
    parser.add_argument('--subset_length', type=int, default=None, required=False, help='Length of subset to test')
        
    
    args = parser.parse_args()
    model_path = args.model
    data_path = args.data_path
    dataset_name = args.dataset
    val_stats_path = args.validation_stats_path
    output_path = args.output_path
    subset_length = args.subset_length

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")

    
    # Load model
    print('loading model')
    ckpt = torch.load(model_path, map_location=device)
    covariateflow = model.create_conditional_flow(device, img_shape=(2,3,32,32) if dataset_name == 'CIFAR10' else (2,3,64,64), train_set=None, num_coupling_layers=8)
    covariateflow.load_state_dict(ckpt['state_dict'])
    covariateflow = covariateflow.eval()

    covariate_transform=transforms.Compose([custom_transform.pil_img_to_numpy, custom_transform.normalize_8bit,
                                             custom_transform.GaussianFilter(1), custom_transform.AdjustHighImage(), custom_transform.toTensor,
                                               custom_transform.ScaleAndQauntizeHigh(bits=16), custom_transform.Permute() ] )

    # Compute val statistics
    if val_stats_path is None:
        if dataset_name == 'CIFAR10':
            dataset = datasets.CIFAR10(root=args.data_path, download=True, transform = covariate_transform, train=True)
        elif dataset_name == 'ImageNet200':
            dataset = TinyImageNet(root=args.data_path, transform = covariate_transform, train=True)

        train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.85), int(len(dataset)*0.15)]) 
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, drop_last=False, num_workers=4)
        print('val loader', len(val_loader))
        val_results = test_dataset(covariateflow, val_loader, device, name = str(dataset_name+'_val'), output_path=args.output_path)
    else:
        val_results = read_validation_stats(val_stats_path)

    # Compute ID test results
    if dataset_name == 'CIFAR10':
        test_set = datasets.CIFAR10(root=args.data_path, download=True, transform = covariate_transform, train=False)
    elif dataset_name == 'ImageNet200':
        test_set = TinyImageNet(root=args.data_path, transform = covariate_transform, train=False)

    if subset_length is not None:
        # use random subset
        test_set = torch.utils.data.Subset(test_set, np.random.choice(len(test_set), subset_length, replace=False))
            
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, drop_last=False, num_workers=4)
    print('test loader', len(test_loader))
    test_results = test_dataset(covariateflow, test_loader, device, val_stats=val_results, name = dataset_name+'_test', output_path=args.output_path)

    # Compute OOD test results
    ood_test_results = {}
    cifar10c_corruptions_list = ['brightness','contrast','defocus_blur','elastic_transform','fog',
                                 'frost','gaussian_blur','gaussian_noise','glass_blur','impulse_noise','jpeg_compression','motion_blur',
                                 'pixelate','saturate','shot_noise','snow','spatter','speckle_noise','zoom_blur']
    
    tinyimagenetc_corruptions_list = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                                       'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    
    if dataset_name == 'CIFAR10':
        
        for corruption in cifar10c_corruptions_list:
            for severity in range(1,6):
                print(f"Corruption: {corruption}, Severity: {severity}")
                cifar10c = CIFAR10C(root=os.path.join(args.data_path, 'CIFAR-10-C'), name=corruption, severity=severity, transform = covariate_transform)
                if subset_length is not None:
                    cifar10c = torch.utils.data.Subset(cifar10c, np.random.choice(len(cifar10c), subset_length, replace=False))
                cifar10c_loader = torch.utils.data.DataLoader(cifar10c, batch_size=1, shuffle=False, drop_last=False, num_workers=4)
        
                cifar10c_results = test_dataset(covariateflow, cifar10c_loader, device, val_stats=val_results, 
                                                name = dataset_name+'_c_'+corruption+'_s_'+str(severity), output_path=args.output_path)
                ood_test_results = compute_results(val_results['mean_bpd'], val_results['std_bpd'], val_results['mean_grad'], val_results['std_grad'], 
                                                   test_results['test_bpd'], test_results['test_grad'], cifar10c_results['test_bpd'], cifar10c_results['test_grad'], 
                                                   corruption, severity, ood_test_results)

    elif dataset_name == 'ImageNet200':

        for corruption in tinyimagenetc_corruptions_list:
            for severity in range(1,6):
                print(f"Corruption: {corruption}, Severity: {severity}")
                tinyimagenetc = TinyImageNetC(root=args.data_path, corruption=corruption, severity=severity, transform = covariate_transform)
                if subset_length is not None:
                    tinyimagenetc = torch.utils.data.Subset(tinyimagenetc, np.random.choice(len(tinyimagenetc), subset_length, replace=False))
                tinyimagenetc_loader = torch.utils.data.DataLoader(tinyimagenetc, batch_size=1, shuffle=False, drop_last=False, num_workers=4)
                tinyimagenetc_results = test_dataset(covariateflow, tinyimagenetc_loader, device, val_stats=val_results, 
                                                     name = dataset_name+'_c_'+corruption+'_s_'+str(severity), output_path=args.output_path)
                ood_test_results = compute_results(val_results['mean_bpd'], val_results['std_bpd'], val_results['mean_grad'], val_results['std_grad'],
                                                    test_results['test_bpd'], test_results['test_grad'], tinyimagenetc_results['test_bpd'], tinyimagenetc_results['test_grad'], 
                                                    corruption, severity, ood_test_results)


    # Compute results
    score_auroc, score_fpr, bpd_auroc, bpd_fpr, grad_auroc, grad_fpr = compute_scores(ood_test_results)
    print(f"NSD AUROC: {score_auroc}, Score FPR: {score_fpr}")
    print(f"BPD AUROC: {bpd_auroc}, BPD FPR: {bpd_fpr}")
    print(f"Grad AUROC: {grad_auroc}, Grad FPR: {grad_fpr}")



