import torch
import open_clip
from sentence_transformers import util
from PIL import Image
import numpy as np
import cv2
from Config import *

# Set device to GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained model and preprocessing function
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
model.to(device)

def imageEncoder(img):
    """
    Encode an image using the pre-trained model.

    Parameters:
        img (np.ndarray): Input image.

    Returns:
        torch.Tensor: Encoded image.
    """
    img = Image.fromarray(img).convert('RGB')
    img = preprocess(img).unsqueeze(0).to(device)
    img = model.encode_image(img)
    return img

def check_misplaced_using_clip(image1, image2, simThresh=0.9):
    """
    Check if two images are similar using CLIP model.

    Parameters:
        image1 (np.ndarray): First input image.
        image2 (np.ndarray): Second input image.
        simThresh (float): Similarity threshold for determining if images are the same.

    Returns:
        bool: True if images are not similar (misplaced), False otherwise.
    """
    img1 = imageEncoder(image1)
    img2 = imageEncoder(image2)

    cos_scores = util.pytorch_cos_sim(img1, img2)
    score = round(float(cos_scores[0][0]) * 100, 2)
    print("score:", score)

    if score > simThresh:
        print("Both are the same images.")
        return False
    else:
        print("Not the same images.")
        return True

def check_misplaced_using_orb(image1, image2):
    """
    Check if two images are similar using ORB feature matching.

    Parameters:
        image1 (np.ndarray): First input image.
        image2 (np.ndarray): Second input image.

    Returns:
        bool: True if images are not similar (misplaced), False otherwise.
    """
    orb = cv2.ORB_create()

    # Convert images to grayscale
    training_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    test_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    # Detect keypoints and descriptors
    train_keypoints, train_descriptor = orb.detectAndCompute(training_gray, None)
    test_keypoints, test_descriptor = orb.detectAndCompute(test_gray, None)

    # Set match threshold
    matchThresh = len(train_keypoints) - 0.1 * len(train_keypoints)
    print("Number of Keypoints Detected In The Training Image:", len(train_keypoints))
    print("Number of Keypoints Detected In The Query Image:", len(test_keypoints))

    # Create a Brute Force Matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(train_descriptor, test_descriptor)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    print(f"Number of Matches: {len(matches)} x {matchThresh}")

    if len(matches) > matchThresh:
        return False
    else:
        return True

def check_misplaced(image1, image2, method="clip", simThresh=0.9):
    """
    Check if two images are similar using the specified method.

    Parameters:
        image1 (np.ndarray): First input image.
        image2 (np.ndarray): Second input image.
        method (str): Method to use for comparison ('clip' or 'orb').
        simThresh (float): Similarity threshold for CLIP method.

    Returns:
        bool: True if images are not similar (misplaced), False otherwise.
    """
    if method == "clip":
        return check_misplaced_using_clip(image1, image2, simThresh)
    elif method == "orb":
        return check_misplaced_using_orb(image1, image2)
    else:
        raise ValueError("Invalid method. Use 'clip' or 'orb'.")
