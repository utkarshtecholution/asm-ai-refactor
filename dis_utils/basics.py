import os
from skimage import io
import torch
import numpy as np
import time


def mae_torch(pred, gt):
    """
    Calculate Mean Absolute Error (MAE) between prediction and ground truth tensors.
    
    Parameters:
        pred (torch.Tensor): Predicted tensor.
        gt (torch.Tensor): Ground truth tensor.

    Returns:
        torch.Tensor: Mean Absolute Error.
    """
    h, w = gt.shape[:2]
    sum_error = torch.sum(torch.abs(pred.float() - gt.float()))
    mae_error = sum_error / (float(h) * float(w) * 255.0 + 1e-4)

    return mae_error


def f1score_torch(pred, gt):
    """
    Calculate precision, recall, and F1 score for the prediction and ground truth tensors.
    
    Parameters:
        pred (torch.Tensor): Predicted tensor.
        gt (torch.Tensor): Ground truth tensor.

    Returns:
        tuple: Tensors containing precision, recall, and F1 score.
    """
    gt_num = torch.sum((gt > 128).float())

    pp = pred[gt > 128]
    nn = pred[gt <= 128]

    pp_hist = torch.histc(pp, bins=255, min=0, max=255)
    nn_hist = torch.histc(nn, bins=255, min=0, max=255)

    pp_hist_flip = torch.flip(pp_hist, dims=[0])
    nn_hist_flip = torch.flip(nn_hist, dims=[0])

    pp_hist_flip_cum = torch.cumsum(pp_hist_flip, dim=0)
    nn_hist_flip_cum = torch.cumsum(nn_hist_flip, dim=0)

    precision = pp_hist_flip_cum / (pp_hist_flip_cum + nn_hist_flip_cum + 1e-4)
    recall = pp_hist_flip_cum / (gt_num + 1e-4)
    f1 = (1 + 0.3) * precision * recall / (0.3 * precision + recall + 1e-4)

    return precision.unsqueeze(0), recall.unsqueeze(0), f1.unsqueeze(0)


def save_prediction_image(pred, save_dir, dataset_name, image_name):
    """
    Save the prediction image to the specified directory.

    Parameters:
        pred (torch.Tensor): Predicted tensor.
        save_dir (str): Directory to save the image.
        dataset_name (str): Name of the dataset.
        image_name (str): Name of the image file.
    """
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        dataset_folder = os.path.join(save_dir, dataset_name)
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
        io.imsave(os.path.join(dataset_folder, f"{image_name}.png"), pred.cpu().numpy().astype(np.uint8))


def f1_mae_torch(pred, gt, valid_dataset, idx, hypar):
    """
    Calculate F1 score, precision, recall, and MAE for the prediction and ground truth tensors.
    Optionally save the prediction image.

    Parameters:
        pred (torch.Tensor): Predicted tensor.
        gt (torch.Tensor): Ground truth tensor.
        valid_dataset (Dataset): Validation dataset containing image information.
        idx (int): Index of the current image in the validation dataset.
        hypar (dict): Hyperparameters including save directory information.

    Returns:
        tuple: Arrays containing precision, recall, F1 score, and MAE.
    """
    start_time = time.time()

    if gt.ndim > 2:
        gt = gt[:, :, 0]

    precision, recall, f1 = f1score_torch(pred, gt)
    mae = mae_torch(pred, gt)

    save_prediction_image(pred, hypar.get("valid_out_dir", ""), valid_dataset.dataset["data_name"][idx], valid_dataset.dataset["im_name"][idx])

    print(f"{valid_dataset.dataset['im_name'][idx]}.png")
    print(f"Time for evaluation: {time.time() - start_time}")

    return precision.cpu().numpy(), recall.cpu().numpy(), f1.cpu().numpy(), mae.cpu().numpy()