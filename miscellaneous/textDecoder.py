import cv2
import os
from ultralytics import YOLO
from functools import reduce
import uuid
import threading 
from threading import Lock, Thread
from google_ocr import cloud_vision_inference
from queue import Queue
import re 
import Config 
from miscellaneous import firebase_update
from torchvision.ops import nms 
import time 
from miscellaneous import send_to_rlef
import traceback 
import math 
import imutils 
import torch
import time 
import shutil 
import numpy as np
from datetime import datetime
from miscellaneous.firebase_update import update_workstation_details

text_localization = YOLO(Config.TEXT_LOCALIZATION_YOLO_PATH)



class TextDecoder:
    def __init__(self,ocr_model_type):
       
        self.ocr_model_type = ocr_model_type
        self.image_path = None
        self.text_model = None
        self.text_localization_yolo = text_localization
        self.image = None
        self.final_outputs = []
        self.clss_map = None
        self.yolo_results = None
        self.ocr_model = None
        # self.yolo_backup = yolo_backupte
        self.thread_lock = Lock()
        self.start_time = None 
        self.end_time = None 
        self.max_thread_count = 10
        self.onboard_flag = False 
        self.compartment_id = None 
        self.best_use_by_confidence = 0
        self.best_ref_no_confidence = 0
        self.best_lot_no_confidence = 0
        self.best_product_name_confidence = 0
        self.cummulative_confidence = 0
        self.unique_id = None
        self.product_image_url = None 
        self.results_queue = Queue()
        self.yolo_time = 0
        self.google_ocr_time = 0
        self.api_calls = 0
        # self.tokenizer = tokenizer
        # self.text_model = model 

        
        self.load_ocr_model()
    
    

    def yolo_segmentation(self, save_img = False, image = None, threshold = 0.5, iou_threshold = 0.5, logging_dir = 'runtimeLog/yolo_results'):
        if image is not None:
            self.image = image 

        results = self.text_localization_yolo(self.image)

        for result in results:
            boxes = result.bboxes.xyxy.tolist()
            class_map = result.names 
            scores = result.bboxes.conf.tolist() 
            classes = result.boxes.cls.tolist() 

            masks = result.masks.xy.tolist()
        
        if len(boxes) == 0:
            self.yolo_results = []
            return None 
        
        ## NMS for removing overlapping bounding boxes 

        keep_indices = nms(boxes, scores, iou_threshold)

        for idx in keep_indices:
            box = boxes[idx]
            box = [int(x) for x in box]
            if scores[idx]> threshold:
                self.yolo_results.append({'xyxy' : box, 
                                          'clss' : class_map[int(classes[idx])], 
                                          'conf' : scores[idx], 
                                          'mask' : masks[idx]
                                         })
        print(f'################# Number of yolo results : {len(self.yolo_results)} #########################')
        

        annotated_image = self.image.copy()
        if save_img:
            if os.path.exists(logging_dir) is False:
                os.mkdir(logging_dir)
            else:
                shutil.rmtree(logging_dir)
                os.mkdir(logging_dir)
            for result in self.yolo_results:
                annotated_image = cv2.bitwise_and(annotated_image, annotated_image, mask = result['mask'])
            cv2.imwrite(os.path.join(logging_dir, 'annotated_image.png'), annotated_image)
        
        self.api_calls = len(self.yolo_results)
        return self.yolo_results 
     



    
    def convert_radians_to_degrees(self, radian):
        degree = math.degrees(radian)
        return degree


        
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
        # send_to_rlef.send_segmentation_to_rlef(uuid.uuid1(), 
        #                                        "backlog", csv="csv", 
        #                                        tag= "us-live-setup", 
        #                                        model="66965ed548169ca1bf32ef13",confidence_score=100,label="catheter",
        #                                        prediction="catheter",model_type="imageAnnotations",
        #                                        file_type="image/png", filename=self.image_path,segmentations=oriented_boxes,confidence_scores=scores,labels=rlef_labels)


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
        


    def polygon_to_mask(self, polygon, image_shape):
        """
        Create a binary mask from a polygon and an image shape.

        Parameters:
        - polygon: List of tuples or list of lists, where each tuple/list is (x, y) coordinates of a polygon vertex.
        - image_shape: Tuple (height, width) representing the shape of the image.

        Returns:
        - mask: A binary mask (numpy array) of the same size as the input image.
        """
        # Initialize a blank mask with the same size as the image
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        
        # Convert polygon points to the required format
        polygon = np.array(polygon, dtype=np.int32)
        polygon = polygon.reshape((-1, 1, 2))

        # Fill the polygon on the mask
        mask = cv2.fillPoly(mask, [polygon], color=255)

        
        return mask
    
    def crop_oriented_bbox_image(self, bbox, oriented_box, angle = None, save_image = False):
        """
            if angle is None, 
                Give the cropped image directly after masking
            else,
                Give two rotated images in the angle given by yolo oriented bounding box angle
        """
        output_image = self.image.copy()
        bbox = [int(x) for x in bbox]
        xmin,ymin,xmax, ymax = bbox 

        mask = self.polygon_to_mask(oriented_box, output_image.shape[:2])
        output_image = cv2.bitwise_and(output_image, output_image, mask = mask)
        cropped_image = output_image[ymin:ymax, xmin:xmax]
        if angle is None:
            if save_image is True:
                image_path = f'runtimeLog/crop/{uuid.uuid1()}.png'
                cv2.imwrite(image_path, cropped_image)
                return image_path 
            else:
                return cropped_image 
        else:
            rot_img_1 = imutils.rotate_bound(cropped_image, angle)
            rot_img_2 = imutils.rotate_bound(cropped_image, angle+180)
            if save_image:
                temp_save_path_1 = f'{Config.OCR_CROP_IMG_PATH}/decode_box_{uuid.uuid1()}.png'
                temp_save_path_2 = f'{Config.OCR_CROP_IMG_PATH}/decode_box_{uuid.uuid1()}.png'
                cv2.imwrite(temp_save_path_1, rot_img_1)
                cv2.imwrite(temp_save_path_2, rot_img_2)
                return temp_save_path_1, temp_save_path_2
            else:
                return rot_img_1, rot_img_2 

        
    def crop_image(self, bbox, angle = None, save_image = False):
        """
            bbox : [xmin,ymin,xmax,ymax]
            oriented_box : [4 coordinates]
            oriented : True if oriented box is given
            angle : angle of the oriented box
 
        """
        output_image = self.image.copy()
        bbox = [int(x) for x in bbox]
        xmin,ymin,xmax,ymax = bbox
        

        ## Direct cropping or mask cropping handled by this if else conditions 
        cropped_image = output_image[ymin:ymax, xmin:xmax]
        if angle is None:
            if save_image:
                temp_save_path = f'{Config.OCR_CROP_IMG_PATH}/decode_box_{uuid.uuid1()}.png'
                cv2.imwrite(temp_save_path, cropped_image)
                return temp_save_path 
            else:
                return cropped_image 
        else:
            rot_img_1 = imutils.rotate_bound(cropped_image, angle)
            rot_img_2 = imutils.rotate_bound(cropped_image, angle+180)
            if save_image:
                temp_save_path_1 = f'{Config.OCR_CROP_IMG_PATH}/decode_box_{uuid.uuid1()}.png'
                temp_save_path_2 = f'{Config.OCR_CROP_IMG_PATH}/decode_box_{uuid.uuid1()}.png'
                cv2.imwrite( temp_save_path_1, rot_img_1)
                cv2.imwrite(temp_save_path_2, rot_img_2)
                return temp_save_path_1, temp_save_path_2
            else:
                return rot_img_1, rot_img_2  

        
    def handle_two_images(self, image_1, image_2):
        self.start_time = time.time()
        image_path_1 = f'runtimeLog/ocr/{uuid.uuid1()}.png'
        image_path_2 = f'runtimeLog/ocr/{uuid.uuid1()}.png'
        cv2.imwrite(image_path_1, image_1)
        cv2.imwrite(image_path_2, image_2)
        results = self.yolo_inference(image = image_1, save_img = True, image_path = image_path_1)
        results_2 = self.yolo_inference(image = image_2, save_img = True, image_path = image_path_2) 
       
        
        if len(results) > len(results_2):
            print('IMAGE 1 is better')
            self.image = image_1 
            
            self.image_path = image_path_1
            self.yolo_results = results 
        else:
            print('IMAGE 2 is better')
            self.image = image_2 
            self.image_path = image_path_2
            self.yolo_results = results_2 

        if self.yolo_results is not None:
            self.final_outputs = []
            threads = []

            for segment in self.yolo_results:
                # print(segment)
                thread = threading.Thread(target = self.process, args = (segment, ))
                threads.append(thread)
                if len(threads) > self.max_thread_count:
                    for th in threads:
                        th.start()
                    for th in threads:
                        th.join() 
                    threads = []
            
            if len(threads) > 0:
                for th in threads:
                    th.start()
                for th in threads:
                    th.join() 
                threads = []

               
    
        return self.restructure_final_output()
        
    

            
            
    def load_ocr_model(self):
        if self.ocr_model_type == 'google-ocr':
            self.ocr_model = cloud_vision_inference
        # elif self.ocr_model_type == 'paddle-ocr':
        #     self.ocr_model = paddle_ocr
        
    def blur_laplace(self, thresh = 100):
        """
        Input: thresh -> Threshold for cv2.Laplacian function. Increase it if your camera quality is very good.
        Decrease it if your camera quality is average.
        """
        
        gray_frame = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
        return int(variance)>=thresh
  
    
    def contains_digit(self, text):
        return any(char.isdigit() for char in text)
    

    
    def text_ocr_process_multi_image(self, img_path1, img_path2, type):
        try:
            result_queue1 = Queue()
            result_queue2 = Queue()
            if os.path.exists(img_path1) is False:
                raise Exception('Cropped image path is not there')
            if os.path.exists(img_path2) is False:
                raise Exception('Cropped image is not there ')
        
            
            thread1 = Thread(target = self.ocr_model, args = (img_path1, result_queue1))
            thread2 = Thread(target = self.ocr_model, args = (img_path2, result_queue2))
            thread1.start()
            thread2.start()
            thread2.join()
            thread1.join()

            result1 = result_queue1.get_nowait()
            result2 = result_queue2.get_nowait()

            confidence_r1 = 0
            confidence_r2 = 0

            text_r1 = ''
            text_r2 = ''

            
            for entry in result1:
                if type == 'barcode':
                    if entry['text'].startswith('('):
                        text_r1 +=" " + entry['text']
                    else:
                        continue 
                else:
                    text_r1 += " " + entry['text']

                confidence_r1 += entry['confidence']
            # return text_r1, img_path1
                
            if len(result1) > 0:
                confidence_r1 /= len(result1)
            else:
                confidence_r1 = 0

            for entry in result2:
                if type == 'barcode':
                    if entry['text'].startswith('('):
                        text_r2 +=" " + entry['text']
                    else:
                        continue 
                else:
                    text_r2  += " " + entry['text']

                confidence_r2 += entry['confidence']
            
            if len(result2) > 0:
                confidence_r2 /= len(result2)
            else:
                confidence_r2 = 0

            if confidence_r1 > confidence_r2:
                os.remove(img_path2)
                return text_r1 , img_path1
            else:
                os.remove(img_path1)
                return text_r2 , img_path2
        except Exception as e:
            print(f'{e} at line 329 in text_decoder.py')
            return None 

            
    def text_ocr_process_single_image(self, img_path, type):
        result = self.ocr_model(img_path)
        text = ""
        ocr_confidence = 0
        for entry in result:
            if type == 'barcode':
                if entry['text'].startswith('('):
                    text +=" " + entry['text']
                else:
                    continue 
            else:
                text += " " + entry['text']

            ocr_confidence += entry['confidence']

        return text 
                


            
        
        
    
    def process(self,segment):
        s = time.time()
        json_output = {'text': None, 
                      'type': segment['clss'], 
                      'xyxy': segment['xyxy'], 
                      'conf': segment['conf'],
                      'crop_img_path': ""
                      }


        ## Handle dual crop if angle correction is not implemented
        # crop_img_path1, crop_img_path2 = self.crop_image(bbox = segment['xyxy'],angle = segment['angle'], oriented_box = segment['obb'] , oriented = True)
        
        # result, crop_img_path = self.text_ocr_process(crop_img_path1, crop_img_path2, json_output['type'])



        # Handle only single image OCR if the image is angle corrected
        crop_img_path = self.crop_oriented_bbox_image(bbox = segment['xyxy'], angle = None,  
                                                      oriented_box= segment['obb'], save_image= True)
        
        result = self.text_ocr_process_single_image(crop_img_path, segment['clss'])


        print(f' RAW TEXT : {result} . TYPE : {segment["clss"]}')

        ## Sending to RLEF ##
        Thread(target = send_to_rlef.send_ocr_image_to_rlef, 
               args = (crop_img_path, result, Config.OCR_MODEL_ID, )).start()
        

        if json_output['type'] == 'lot_no':
            result = result.lower()
            result = result.replace('lot', '')
            result = result.replace('number', '')
            result = result.replace('catalogue', '')
            result = result.replace('catalog', '')
            result = result.replace('batch', '')
            result = result.replace('code', '')
            result = result.replace(':', " ")
            result = result.upper()
            # result = self.filter_raw_text(result, 'LOT')
            result = self.extract_number(result, 'LOT')

        elif json_output['type'] == 'ref_no':
            result = result.lower()
            result = result.replace('reference', '')
            result = result.replace('batch', '')
            result = result.replace('code', '')
            result = result.replace(':', '')
            result = result.replace('number', '')
            result = result.replace('ref', '')
            result = result.replace('rep', '')
            result = result.replace('catalogue' ,'')
            result = result.replace('catalog', '')
            result = result.replace(':', " ")
            result = result.upper()
            result = self.extract_number(result, 'REF')
            # result = self.filter_raw_text(result, 'REF')
         
            
        elif json_output['type'] == 'use_by':
            result = self.extract_number(result, 'USE_BY')

        else:
            result = result
        json_output['crop_img_path'] = crop_img_path 
        # print(f">>>>> TYPE : {segment['clss']} . TEXT : {result} <<<<<<<<")
        # firebase_update.dynamic_update_onboardin, crop_img_path, json_output['type'], result)

        if result is None:
            json_output['text'] = None 
        else:
            json_output['text'] = result 

        print(f'REFINED TEXT : {result}. TYPE : {segment["clss"]}. SCORE: {segment["conf"]}')
        self.final_outputs.append(json_output)
        e = time.time()
        self.google_ocr_time = e-s

        # os.remove(crop_img_path)
        
        
    
    
    def decode_image_text(self, image):
        
        
        self.start_time = time.time()
        # print(f'>>>>>> Image path at decoding image : {img_path} <<<<<<<<<<<')
        # self.image_path = img_path
        # if os.path.exists(self.image_path) is not True:
        #     raise Exception(f'Image path does not exist')

        self.image = image 
        
        
        yolo_st = time.time()
        self.yolo_obb_inference(threshold=Config.YOLO_THRESHOLD, save_img=True)
        yolo_et = time.time()
        self.yolo_time = yolo_et - yolo_st 



        if self.yolo_results is not None:
            self.final_outputs = []
            threads = []

            for segment in self.yolo_results:
            
                thread = threading.Thread(target = self.process, args = (segment, ))
                threads.append(thread)
                if len(threads) > self.max_thread_count:
                    for th in threads:
                        th.start()
                    for th in threads:
                        th.join() 
                    threads = []
            
            if len(threads) > 0:
                for th in threads:
                    th.start()
                for th in threads:
                    th.join() 
                threads = []

               
    
        return self.restructure_final_output()


    def digit_exists_in_text(self, text):
        return any(char.isdigit() for char in text)
        
    



    def extract_number(self, ocr_text, number_type):
        """
        Extract the specified number (e.g., LOT number or REF number) from the OCR text.
        The number should start after the specified type (e.g., 'LOT' or 'REF') and begin with a character between A-Z or 0-9.

        Parameters:
        ocr_text (str): The OCR text from which to extract the number.
        number_type (str): The type of number to extract ('LOT' or 'REF').

        Returns:
        str: The extracted number, or an empty string if no valid number is found.
        """
        # Define regex patterns for different number types
        patterns  = {
            'USE_BY': r'(\d{4}-\d{2}-\d{2})',
            'REF': r'\b(?:[A-Za-z]*\d+[A-Za-z]*)(?:-[A-Za-z]*\d+[A-Za-z]*)*\b',
            'LOT': r'\b(?:[A-Za-z]*\d+[A-Za-z]*)(?:-[A-Za-z]*\d+[A-Za-z]*)*\b'
        }

        # Validate number_type and get the corresponding pattern
        if number_type not in patterns:
            raise ValueError(f"Unsupported number type: {number_type}")

        if number_type == 'REF' or number_type == 'LOT':
            final_text = ""
            ocr_text = ocr_text.split(' ')
            print(f" SPLIT {ocr_text}")
            ## To be changed later
            if len(ocr_text) > 1:
                for word in ocr_text:
                    if self.digit_exists_in_text(word):
                        word = word.replace(' ', '')
                        final_text += " " + word 

                return final_text 
                

            else:
                ocr_text = ocr_text[0]
                ocr_text = ocr_text.replace(' ', '')
                return ocr_text
                # try:
                #     matches = re.findall(patterns[number_type], ocr_text)
                #     valid_matches = [match for match in matches if any(c.isdigit() for c in match) and any(c.isalpha() for c in match)]
                #     return str(valid_matches[0])
                # except Exception as e:
                #     print(e)
                #     return ocr_text
        elif number_type == 'USE_BY':

            pattern = patterns[number_type]
            
            # Compile the pattern and search for matches
            number_pattern = re.compile(pattern, re.IGNORECASE)
            match = number_pattern.search(ocr_text)

            if match:
                # Return the extracted number
                return match.group(0)
            else:
                # Return an empty string if no valid number is found
                return ''

    def extract_and_format_date(self, barcode_text):
    # Extract the date part between (17) and (10)
        date_match = re.search(r'\(17\)(\d{6})', barcode_text)
        
        if date_match:
            # Extract the date part
            date_str = date_match.group(1)
            # Format the date part to YYYY-MM-DD
            formatted_date = f'20{date_str[:2]}-{date_str[2:4]}-{date_str[4:]}'
            return formatted_date
        else:
            return ""
        
    def clear_previous_outputs(self):
        self.best_lot_no_confidence = 0
        self.best_product_name_confidence = 0
        self.best_ref_no_confidence = 0
        self.best_use_by_confidence = 0
        self.yolo_results = None 
        self.final_outputs = [] 
        self.results_queue = None 



    def get_image_url_parallelized(self,image_paths):
        signed_urls = {}
        threads = []
        for entry in image_paths.keys():
            if image_paths[entry] == '':
                signed_urls[entry] = ''
                continue
            
            thread =  Thread(target = firebase_update.upload_and_generate_signed_url, args = (image_paths[entry], self.results_queue, entry, True))
            threads.append(thread)

        for th in threads:
            th.start()
        for th in threads:
            th.join()


        while True:
            try:
                result = self.results_queue.get_nowait()
                signed_urls[result["type"]] = result['url']
            except Exception as e:
                print(f'{e} at line 741 in text-decoder.py')
                break 

      
       
        
   
        return signed_urls

    def remove_runtime_image_paths(self, path_list):
        for path in path_list:
            os.remove(path)
        
   

    def restructure_final_output(self):
        '''
            Out of all the barcodes and segments created, choose the best.
            barcode (str) : Longest Barcode 
            ref_no (str) : Reference number with the highest confidence score 
            lot_no (str) : Lot Number with the highest confidence score 
            use_by (str) : Give the text of use_by date. 

        '''
        final_barcode = '' 
        final_ref_no = '' 
        final_lot_no = ''  
        final_use_by = ''  
        final_product_name = '' 
        image_paths = {
            "barcode" : "", 
            "ref_no" : "", 
            "lot_no" : "" , 
            "use_by" : "",
            "product_image" : self.image_path
        }
       
        
        final_serial_number = 'NA'
        # print(self.final_outputs)

        

        if len(self.final_outputs) > 0:
            for entry in self.final_outputs:
                try:
                    if entry['type'] == 'barcode':
                        if len(entry['text']) > len(final_barcode):
                            final_barcode = entry['text']
                            final_barcode = final_barcode.replace(' ', '')
                            image_paths['barcode'] = entry['crop_img_path']
                        # else:
                        #     os.remove(entry['crop_img_path'])
                            # final_barcode_image_path = firebase_update.upload_and_generate_signed_url(entry['crop_img_path'], None)
                    if entry['type'] == 'ref_no':
                        if entry['conf'] > self.best_ref_no_confidence:
                            self.best_ref_no_confidence = entry['conf']
                            final_ref_no = entry['text']
                            image_paths['ref_no'] = entry['crop_img_path']
                        # else:
                        #     os.remove(entry['crop_img_path'])
                            # final_ref_image= firebase_update.upload_and_generate_signed_url(entry['crop_img_path'], None )
                    if entry['type'] == 'lot_no':
                        if entry['conf'] > self.best_lot_no_confidence:
                            final_lot_no = entry['text']
                            self.best_lot_no_confidence = entry['conf']
                            # final_lot_path = firebase_update.upload_and_generate_signed_url(entry['crop_img_path'], None)
                            image_paths['lot_no'] = entry['crop_img_path']
                        # else:
                        #     os.remove(entry['crop_img_path'])

                    if entry['type'] == 'use_by':
                        if  entry['conf'] > self.best_use_by_confidence:
                            final_use_by = entry['text']
                            self.best_use_by_confidence = entry['conf']
                            # final_use_by_image= firebase_update.upload_and_generate_signed_url(entry['crop_img_path'], None )
                            image_paths['use_by'] = entry['crop_img_path']
                        # else:
                        #     os.remove(entry['crop_img_path'])

                   
                except Exception as e:
                    print(f'>>>>>>>>>>>>>>>> {e} <<<<<<<<<<<<<<<<<<<<<')
                    traceback.print_exc()
                    continue
        
        



        
        if final_barcode != '':
            if self.extract_and_format_date(final_barcode) != '':
                final_use_by = self.extract_and_format_date(final_barcode)
            
        if '(21)' in final_barcode:
            final_serial_number = final_barcode.split('(21)')[-1]
        
        if '(10)' in final_barcode:
            final_lot_no = final_barcode.split('(10)')[-1]
            try:
                final_lot_no = final_lot_no.split('(')[0]
            except:
                print("it don't have any brackets after the lot number")
            

        signed_urls = self.get_image_url_parallelized(image_paths)
        print('FINAL OUTPUT')
        restructured_output: dict = {
            'ref_no': final_ref_no.replace(' ', ''), 
            'lot_no' : final_lot_no.replace(' ', ''), 
            'use_by' : str(int(time.mktime(datetime.strptime(final_use_by, "%Y-%m-%d").timetuple()))*1000),
            'barcode' : final_barcode,
            'captured_barcode': signed_urls['barcode'] , 
            'captured_lot_no' : signed_urls['lot_no']  , 
            'captured_ref_no' : signed_urls['ref_no']  , 
            'captured_use_by' : signed_urls['use_by'] , 
            'captured_image' : signed_urls['product_image'],
            'product_image' : signed_urls['product_image'],
            'serial_no' : final_serial_number,
            'is_documented' : False
        }
        
        sku_details = update_workstation_details(restructured_output)

        print(sku_details)

        firebase_update.update_workstation_details(sku_details)
        
        self.end_time = time.time()
        print(f'>>>>>>> TIME TAKEN FOR OCR : {self.end_time - self.start_time} <<<<<<<<<<<<<<')

        self.clear_previous_outputs()
        return restructured_output
