from dis_utils import dis_inference
import Config
from utils.maskHelper import MaskImage
import cv2

## loading the dis model hyperparameter####
dis_inference.load_hyperparameters("weights", Config.DIS_MODEL_PATH)

# Creating Mask Object
mask_helper = MaskImage()

def dis_object_seg(frame):
    
    mask = dis_inference.inference(frame)

    if mask_helper.get_xyxy_mask(mask) is not None:
        return mask
    return None
