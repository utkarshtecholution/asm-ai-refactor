import cv2 
from utils import maskHelper, angle_correction_helper
import imutils 
from queue import Queue 
from ai_pipeline.label_detection import YOLOInference
import traceback

mask_helper = maskHelper.MaskImage()
angle_correction = angle_correction_helper.ImageOrientationCorrection()

yolo_inference = YOLOInference()

def get_best_rotated_image(image, mask, possible_angles):
    """
        image: np.array 
        mask : np.array
        possible_angles: list of angles

        Note:
        image.shape[:2] should be equal to mask.shape
    """

    if image.shape[:2] != mask.shape:
        raise ValueError(f"Image shape ({image.shape[:2]}) and mask {mask.shape[:2]} shape do not match")
    
    # TODO: Parallarlising the YOLO best results logic
    best_image = None 
    best_mask = None 
    best_results = 0
    for idx, angle in enumerate(possible_angles):
        rotated_image = imutils.rotate_bound(image, angle)
        rotated_mask = imutils.rotate_bound(mask, angle)
        results = yolo_inference.yolo_obb_inference(save_img=False, image = rotated_image, mask = rotated_mask, resize=True)
        
        # Checking which image has maximum numbers of label detected or length of results
        if len(results)>best_results:
            best_image = rotated_image
            best_mask = rotated_mask
            best_results = len(results)

    return best_image, best_mask 

def angle_correction(high_res_frame, low_res_mask):
    try:
        # firebase_update.update_status_workstation_ui(str(uuid.uuid1()))
        #TODO: Need to integrate MQTT here for status updates
        high_res_mask = cv2.resize(low_res_mask, high_res_frame.shape[:2][::-1], cv2.INTER_CUBIC)
        
        if high_res_frame.shape[:2] != high_res_mask.shape:
            print(f'The resized mask shape is {high_res_mask.shape}')
            print(f'The image shape is {high_res_frame.shape}')
            raise Exception("Both Image and Masks are not of same size")

        # Geting all four angles
        all_angles_list = angle_correction.all_angles(high_res_mask, high_res_frame)
        
        # Finding best angle corrected image using YOLO
        best_image, best_mask = get_best_rotated_image(high_res_frame, high_res_mask, all_angles_list)
        
        # Cropping Image using mask
        cropped_image = mask_helper.cropped_image(best_image, best_mask)  

        #TODO: scope of optimisation using same best result from YOLO for label detection and croping 
        results = yolo_inference.yolo_obb_inference(image=cropped_image)
        return results

    except:
        traceback.print_exc()
        return 
