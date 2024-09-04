import cv2
import uuid
import Config
import imutils
import numpy as np

def crop_oriented_bbox_image(image, bbox, oriented_box, angle = None, save_image = False):
    """
        if angle is None, 
            Give the cropped image directly after masking
        else,
            Give two rotated images in the angle given by yolo oriented bounding box angle
    """
    output_image = image.copy()
    bbox = [int(x) for x in bbox]
    xmin,ymin,xmax, ymax = bbox 

    mask = polygon_to_mask(oriented_box, output_image.shape[:2])
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

def polygon_to_mask(polygon, image_shape):
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
    