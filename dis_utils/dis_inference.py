import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

# Add the script directory to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
import random
# Import custom modules
from data_loader_cache import normalize, im_preprocess
from models import ISNetDIS
from utils.maskHelper import MaskImage
# from rlef_utils import send_to_rlef

# Global variables
hypar = {}
net = None
masker = MaskImage()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GOSNormalize(object):
    """
    Normalize the Image using torch.transforms.
    """
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return normalize(image, self.mean, self.std)

# Define the transformation pipeline
transform = transforms.Compose([GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])])

def load_image(im, hypar):
    """
    Load and preprocess an image.

    Parameters:
        im (np.array): Input image array.
        hypar (dict): Hyperparameters dictionary.

    Returns:
        tuple: Transformed image tensor and its original shape tensor.
    """
    im, im_shp = im_preprocess(im, hypar["cache_size"])
    im = torch.divide(im, 255.0)
    shape = torch.from_numpy(np.array(im_shp))
    return transform(im).unsqueeze(0), shape.unsqueeze(0)

def build_model(hypar, device):
    """
    Build and load the model with the specified hyperparameters.

    Parameters:
        hypar (dict): Hyperparameters dictionary.
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        nn.Module: The loaded model.
    """
    global net
    net = hypar["model"]

    # Convert to half precision if specified
    if hypar["model_digit"] == "half":
        net.half()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    net.to(device)

    # Load the pre-trained model weights
    if hypar["restore_model"]:
        model_path = os.path.join(hypar["model_path"], hypar["restore_model"])
        net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    return net

def predict(inputs_val, shapes_val, hypar, device):
    """
    Given an image, predict the mask.

    Parameters:
        inputs_val (torch.Tensor): Preprocessed input image tensor.
        shapes_val (torch.Tensor): Original shape tensor of the image.
        hypar (dict): Hyperparameters dictionary.
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        np.array: Predicted mask as a numpy array.
    """
    global net
    net.eval()

    # Set input tensor type based on model precision
    if hypar["model_digit"] == "full":
        inputs_val = inputs_val.type(torch.FloatTensor)
    else:
        inputs_val = inputs_val.type(torch.HalfTensor)

    inputs_val_v = Variable(inputs_val, requires_grad=False).to(device)
    ds_val = net(inputs_val_v)[0]
    pred_val = ds_val[0][0, :, :, :]

    # Resize prediction to original image size
    pred_val = torch.squeeze(F.interpolate(torch.unsqueeze(pred_val, 0), size=(shapes_val[0][0], shapes_val[0][1]), mode='bilinear'))
    pred_val = (pred_val - torch.min(pred_val)) / (torch.max(pred_val) - torch.min(pred_val))

    if device == 'cuda':
        torch.cuda.empty_cache()

    return (pred_val.detach().cpu().numpy() * 255).astype(np.uint8)

def load_hyperparameters(model_dir, weights_path):
    """
    Load hyperparameters and build the model.

    Parameters:
        model_dir (str): Directory containing the model.
        weights_path (str): Path to the model weights file.
    """
    global hypar
    hypar["interm_sup"] = False
    hypar["model_digit"] = "full"
    hypar["seed"] = 0
    hypar["cache_size"] = [640, 640]
    hypar["input_size"] = [640, 640]
    hypar["crop_size"] = [640, 640]
    hypar["model_path"] = model_dir
    hypar["restore_model"] = weights_path
    hypar["model"] = ISNetDIS()

    build_model(hypar, device)
    print('########### DIS MODEL LOADED ################')

def inference(image, threshold_value=240, max_value=255):
    """
    Perform inference on the given image to predict the mask.

    Parameters:
        image (np.array): Input image array.
        threshold_value (int): Threshold value for binarization.
        max_value (int): Maximum value for binarization.

    Returns:
        np.array: Binary mask of the input image.
    """
    image_tensor, orig_size = load_image(image, hypar)
    mask = predict(image_tensor, orig_size, hypar, device)
    _, binary_image = cv2.threshold(mask, threshold_value, max_value, cv2.THRESH_BINARY)


    image_points = masker.generatePointsFromSegment(binary_image)
    image_points = image_points[::10]
    # num = random.randint(0, 1000)
    # cv2.imwrite(f"{num}.png", image)
    # send_to_rlef.send_segmentation_to_rlef(f"{random.randint(0, 1000)}", 
    #                                            "backlog", csv="csv", 
    #                                            tag= "dis-segmentation-live", 
    #                                            model="66a22715cc437fd264210696",confidence_score=100,label="catheter",
    #                                            prediction="catheter",model_type="imageAnnotations",
    #                                            file_type="image/png", filename=f"{num}.png",segmentations=[image_points],confidence_scores=[100],labels=["object"])


    return binary_image