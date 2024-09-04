import cv2 
import uuid 
import Config
from dis_utils import dis_inference
from utils import maskHelper, angle_correction_helper, image_utils
from ocr_utils.textDecoder import TextDecoder
import time 
# from db_utils import firebase_update
import uuid 
import imutils 
from queue import Queue 
from concurrent.futures import ThreadPoolExecutor
from utils import remove_boarder
import numpy as np
import os
from torchvision.ops import nms

## loading the dis model hyperparameter####
dis_inference.load_hyperparameters("weights", Config.DIS_MODEL_PATH)
threads_executor = ThreadPoolExecutor(max_workers=5)

mask_helper = maskHelper.MaskImage()
angle_correction = angle_correction_helper.ImageOrientationCorrection()

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
    
    best_image = None 
    best_mask = None 
    best_results = 0 
    threads = []
    results_queue = Queue(maxsize = 8)
    text_decoder = TextDecoder(ocr_model_type="google-ocr")
    for idx, angle in enumerate(possible_angles):
        rotated_image = imutils.rotate_bound(image, angle)
        rotated_mask = imutils.rotate_bound(mask, angle)
        text_decoder.yolo_obb_inference(queue_object= results_queue, save_img= True, image = rotated_image, mask = rotated_mask, resize=True)
        
    for idx in range(len(possible_angles)):
        try:
            result = results_queue.get_nowait()
        except Exception as e:
            print(f'{e} at line 77 in ocr_service.py')
            continue 

        if result is None:
            continue

        if len(result['yolo_results']) > best_results:
            best_image = result['image']
            best_mask = result['mask']
            best_results = len(result['yolo_results'])

    return best_image, best_mask 



def process_image_ocr(high_res_frame):
    """
    Processes an image frame using OCR (Optical Character Recognition) and associated image processing techniques.

    This function performs several steps to extract text from an image frame, including image preprocessing, 
    border removal, mask generation, angle correction, and OCR. It also measures and logs the time taken 
    for each step. The function saves intermediate and final images for debugging and validation purposes, 
    and updates a CSV file with timing data for performance profiling.

    Args:
        high_res_frame (numpy.ndarray): The high-resolution image frame to be processed. This should be 
                                         a 3-dimensional numpy array (height x width x channels).

    Returns:
        dict or None: Returns a dictionary containing extracted SKU details if text is successfully detected. 
                      Returns None if no text is detected or if an error occurs.

    Raises:
        Exception: Raises an exception if the resized mask and the original image do not have the same dimensions.

    Side Effects:
        - Saves images to disk for testing and logging purposes.
        - Updates a CSV file (Config.TIME_PROFILE_PATH) with timing information for profiling the various stages of processing.
        - Logs detailed timing information and error messages to the console.

    Notes:
        - Assumes the existence of external functions and modules such as `remove_boarder`, `dis_inference`, 
          `angle_correction`, `get_best_rotated_image`, `mask_helper`, `TextDecoder`, and `image_utils`.
        - Uses threading to save images asynchronously.
        - Relies on `firebase_update` for status updates and `uuid` for generating unique filenames.
    """

    ## Check if object exists or not 
    # dis_st = time.time()
    # mask = dis_inference.inference(cv2.resize(high_res_frame, Config.LOW_RES_SHAPE[::-1],  cv2.INTER_LINEAR))
    # dis_et = time.time()
    status, mask = dis_object_detection(high_res_frame)
    if status is True:
        try:
            # firebase_update.update_status_workstation_ui(str(uuid.uuid1()))
            #TODO: Need to integrate MQTT here for status updates

            # text_decoder = TextDecoder(ocr_model_type="google-ocr")
            # s = time.time()
            # print('>>>>> ONBOARDING OCR <<<<<<<<<')

            ## Saving the Runtime image for testing purposes

            # threads_executor.submit(image_utils.save_image, f"runtimeLog/testing_images/{uuid.uuid1()}.png", high_res_frame)
            
            # image_path = f'runtimeLog/crop/{uuid.uuid1()}.png'

            high_res_frame = remove_boarder.remove_border_pixels(high_res_frame, border_size=4)


            # dis_st = time.time()
            # mask = dis_inference.inference(cv2.resize(high_res_frame, Config.LOW_RES_SHAPE[::-1],  cv2.INTER_LINEAR))
            # dis_et = time.time()
            
            # print(f'Resizing image and dis inferencing takes : {dis_et-dis_st}')

            resized_mask = cv2.resize(mask, high_res_frame.shape[:2][::-1], cv2.INTER_CUBIC)
            
            if high_res_frame.shape[:2] != resized_mask.shape:
                print(f'The resized mask shape is {resized_mask.shape}')
                print(f'The image shape is {high_res_frame.shape}')
                raise Exception("Both Image and Masks are not of same size, check line 32 in process.py")
            
            # cv2.imwrite('runtimeLog/dis_mask.png', resized_mask) ## Just for logging purpose for validating DIS inference

            threads_executor.submit(image_utils.save_image, 'runtimeLog/dis_mask.png', resized_mask) 

            angle_st = time.time()
            all_angles_list = angle_correction.all_angles(resized_mask, high_res_frame)
            angle_et = time.time()
            angle_prediction_time = angle_et - angle_st 

            print(f'{angle_et - angle_st} for predicting the correct angle orientation')
        
            best_st = time.time()
            best_image, best_mask = get_best_rotated_image(high_res_frame, resized_mask, all_angles_list)
            best_et = time.time()
            angle_correction_time = best_et - best_st 
            print(f'>>>>>>>>  {best_et - best_st} for getting the best frame <<<<<<<<<<')

            
            cropped_image = mask_helper.cropped_image(best_image, best_mask) 

            return cropped_image

        except:
            pass

    def yolo_obb_inference(self, queue_object = None, save_img = False ,image = None,mask = None, threshold = 0.5, iou_threshold = 0.5,
                                logging_dir = 'runtimeLog/yolo_results', resize = False ):

            if image is not None:
                self.image = image
                
            if self.image is None:
                raise ValueError('>>>>> Image still not initiated <<<<<<<<<<')

            if resize:
                resized_image = cv2.resize(self.image, (640, 480), cv2.INTER_LINEAR)
                results = self.text_localization_yolo(resized_image) 
            else:
                results = self.text_localization_yolo(self.image)

            self.yolo_results = []

            rlef_labels = []

            for result in results:
                class_map = result.names
                boxes = result.obb.xyxy.tolist()
                oriented_bbox = result.obb.xyxyxyxy.tolist()
                scores = result.obb.conf.tolist()
                boxes = result.obb.xyxy
                scores = result.obb.conf
                classes = result.obb.cls.tolist()
                boxes_with_angles = result.obb.xywhr.tolist()
                oriented_boxes = result.obb.xyxyxyxy.tolist()
                # print(oriented_boxes)

            if len(boxes) == 0:
                self.yolo_results = []
                if queue_object is not None:
                    queue_object.put(None)
                return  
            
            keep_indices = nms(boxes, scores, iou_threshold)
            
            rlef_boxes = oriented_bbox
            rlef_labels = []
            confidence_scores_rlef = []

            ## Non Max Suppression for removing overlapping boxes
            for idx in keep_indices:
                box = boxes[idx]
                box = [int(x) for x in box]
                
                rlef_labels.append(class_map[int(classes[idx])])
                confidence_scores_rlef.append(scores[idx])
                if scores[idx]> threshold:
                    self.yolo_results.append({'xyxy' : box, 
                                            'clss' : class_map[int(classes[idx])], 
                                            'conf' : scores[idx], 
                                            'angle' : self.convert_radians_to_degrees(boxes_with_angles[idx][-1]), 
                                            'obb' : oriented_boxes[idx]
                                            })
            print(f'################# Number of yolo results : {len(self.yolo_results)} #########################')

            if queue_object is not None:

                queue_object.put( {
                                    "yolo_results" : self.yolo_results,
                                    "image" : self.image, 
                                    "mask" : mask, 
                                }
                                )
            # ## Integrating RLEF 
            # send_to_rlef.send_segmentation_to_rlef(uuid.uuid1(), "backlog", "csv", Config.TEXT_DETECTION_MODEL_ID, label = "live-setup-us", 
            #                                        confidence_score=100, prediction = "predicted", model_type = "imageAnnotations", 
            #                                        filename = self.image_path, file_type="png", segmentations = rlef_boxes, 
            #                                        confidence_scores=confidence_scores_rlef, labels = rlef_labels, tag = "us-live")
        

            self.api_calls = len(self.yolo_results)
        
            if save_img:
                annotated_image = self.image.copy() 
                for polygon in oriented_boxes:
                    polygon = np.array(polygon, dtype = np.int32)
                    points = polygon.reshape((-1, 1, 2))
                    
                    annotated_image = cv2.fillPoly(annotated_image, [points], color=(255, 0, 0))
            
                cv2.imwrite(os.path.join(logging_dir, 'annotated_image.png'), annotated_image)

            return self.yolo_results