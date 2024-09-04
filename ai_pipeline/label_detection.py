from ultralytics import YOLO 
from torchvision.ops import nms
import Config
import cv2 

text_localization = YOLO(Config.TEXT_LOCALIZATION_YOLO_PATH)

class YOLOInference:
    def __init__(self):
        self.image_path = None
        self.text_model = None
        self.text_localization_yolo = text_localization
        self.image = None
        self.final_outputs = []
        self.clss_map = None
        self.yolo_results = None    

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
                
        return self.yolo_results 
     
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
             
        return self.yolo_results
