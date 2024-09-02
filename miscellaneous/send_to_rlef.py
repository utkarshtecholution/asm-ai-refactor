"""
This script contains utility functions to send data to AutoAI's RLEF endpoint.
"""
import requests
import random
import string
import json
import shutil
import os
import threading
from miscellaneous import rlef_helper as rlh 
import cv2 
import uuid 
import Config 
# Constants
RLEF_URL = "https://autoai-backend-exjsxe2nda-uc.a.run.app/resource/"
HIGH_CONTRAST_COLORS = [
    'rgba(0,255,81,1)', 'rgba(255,219,0,1)', 'rgba(255,0,0,1)', 'rgba(0,4,255,1)', 'rgba(227,0,255,1)'
]
SKIP_ANNOTATIONS = 1  # If 70, we will only use every 70th point

def generate_random_id(digit_count):
    """
    Generate a random string of lowercase letters and digits.

    Parameters:
        digit_count (int): Number of characters in the generated string.

    Returns:
        str: Randomly generated string.
    """
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(digit_count))

def mask_points_to_format(masks, confidence_scores, labels, is_closed=False):
    """
    Format mask points into the required RLEF format.

    Parameters:
        masks (list): List of mask points.
        confidence_scores (list): List of confidence scores.
        labels (list): List of labels.
        is_closed (bool): Whether the masks are closed shapes.

    Returns:
        list: Formatted data.
    """
    formatted_data = []

    for mask_points_list, confidence_score, label in zip(masks, confidence_scores, labels):
        random_id_first = generate_random_id(8)
        vertices = []

        for mask_index, mask_points in enumerate(mask_points_list):
            if mask_index % SKIP_ANNOTATIONS != 0:
                continue

            x, y = mask_points
            vertex_id = random_id_first if len(vertices) == 0 else generate_random_id(8)

            vertex = {
                "id": vertex_id,
                "name": vertex_id,
                "x": int(x),
                "y": int(y),
            }
            vertices.append(vertex)

        mask_data = {
            "id": random_id_first,
            "name": random_id_first,
            "color": random.choice(HIGH_CONTRAST_COLORS),
            "isClosed": is_closed,
            "vertices": vertices,
            "confidenceScore": int(confidence_score * 100),
            "selectedOptions": [
                {"id": "0", "value": "root"},
                {"id": random_id_first, "value": label}
            ]
        }

        formatted_data.append(mask_data)

    return formatted_data

def send_to_autoai(unique_id, status, csv, model, label, tag, confidence_score, prediction, model_type, filename, image_annotations, file_type="image/png", delete_flag = False ):
    """
    Send data to AutoAI's RLEF endpoint.

    Parameters:
        unique_id (str): Unique identifier.
        status (str): Status of the data.
        csv (str): CSV data.
        model (str): Model information.
        label (str): Label information.
        tag (str): Tag information.
        confidence_score (float): Confidence score.
        prediction (str): Prediction information.
        model_type (str): Model type.
        filename (str): Path to the file to be sent.
        image_annotations (str): Image annotations in JSON format.
        file_type (str): Type of the file.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        new_filename = f"runtimeLog/onboarding/{uuid.uuid1()}.png"
        shutil.copy(filename, new_filename)

        payload = {
            'status': status,
            'csv': csv,
            'model': model,
            'label': label,
            'tag': tag,
            'confidence_score': confidence_score,
            'prediction': prediction,
            'imageAnnotations': image_annotations,
            'model_type': model_type
        }

        files = [('resource', (new_filename, open(new_filename, 'rb'), file_type))]
        headers = {}
        response = requests.post(RLEF_URL, headers=headers, data=payload, files=files, verify=False)
        # print(response.txt)


        if response.status_code == 200:
            print('Successfully sent to AutoAI', end="\r")
            if delete_flag:
                os.remove(new_filename)
            return True
        else:
            print('Error while sending to AutoAI')
            print(response.text)
            print(response.status_code)
            return False

    except Exception as e:
        print('Error while sending data to AutoAI:', e)
        return False

def send_image_to_rlef(unique_id, status, csv, model, label, tag, confidence_score, prediction, model_type, filename, file_type, delete_flag= True):
    """
    Send image data to RLEF endpoint.

    Parameters:
        unique_id (str): Unique identifier.
        status (str): Status of the data.
        csv (str): CSV data.
        model (str): Model information.
        label (str): Label information.
        tag (str): Tag information.
        confidence_score (float): Confidence score.
        prediction (str): Prediction information.
        model_type (str): Model type.
        filename (str): Path to the file to be sent.
        file_type (str): Type of the file.
    """
    threading.Thread(target=send_to_autoai,
                     args=(unique_id, status, csv, model, label, tag, confidence_score, prediction, model_type,
                           filename, "[]", file_type, delete_flag)).start()

def send_segmentation_to_rlef(unique_id, status, csv, model, label, tag, confidence_score, prediction, model_type, filename, file_type, segmentations, confidence_scores, labels):
    """
    Send segmentation data to RLEF endpoint.

    Parameters:
        unique_id (str): Unique identifier.
        status (str): Status of the data.
        csv (str): CSV data.
        model (str): Model information.
        label (str): Label information.
        tag (str): Tag information.
        confidence_score (float): Confidence score.
        prediction (str): Prediction information.
        model_type (str): Model type.
        filename (str): Path to the file to be sent.
        file_type (str): Type of the file.
        segmentations (list): List of segmentation points.
        confidence_scores (list): List of confidence scores for each segmentation.
        labels (list): List of labels for each segmentation.
    """
    image_annotations = mask_points_to_format(segmentations, confidence_scores, labels, is_closed=True)
    threading.Thread(target=send_to_autoai,
                     args=(unique_id, status, csv, model, label, tag, confidence_score, prediction, model_type,
                           filename, json.dumps(image_annotations), file_type)).start()
    


def send_dis_segment_to_rlef(image_rgb, points, model_id):
    image_path = f'runtimeLog/DIS/{uuid.uuid1()}.jpg'
    cv2.imwrite(image_path, image_rgb)

    
    rlef_annotations = rlh.segmentation_annotation_rlef(points, "object")

    rlh.send_to_rlef(status = "backlog", img_path=image_path, model_id = model_id, tag = Config.US_TAG, label = "object", 
                        annotation=rlef_annotations)
    os.remove(image_path)


def send_ocr_image_to_rlef(image_path, text, model_id):
    rlh.send_to_rlef(status = "backlog", img_path=image_path, model_id=model_id, tag = Config.US_TAG, label = str(text), annotation = None)



def sending_videos(label, filename, model_id, tag, type = 'mp4'):
    url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/resource"
    payload = {
        'model': model_id,
        'status': 'backlog',
        'csv': 'csv',
        'label': label,
        'tag': tag,
        'model_type': 'video',
        'prediction': label,
        'confidence_score': '100',
        'appShouldNotUploadResourceFileToGCS': 'true',
        'resourceFileName': filename,
        'resourceContentType': "video/mp4"
    }
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload)

    headers = {'Content-Type': f'video/{type}'}

    print(response.status_code)

    api_url_upload = response.json()["resourceFileSignedUrlForUpload"]

    response = requests.request("PUT", api_url_upload, headers=headers, data=open(f"{filename}", 'rb'))
    os.remove(filename)

    
     

def sending_images(label,filename,model, tag):
  print(filename)

  url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/resource"
  try:
      payload = {'status': "backlog",
                'csv': "",
                'model': model,
                'label': label,
                'tag': tag,
                'confidence_score': "100",
                'prediction': "rgb",
                'imageAnnotations': {},
                'model_type': "image/png"}
      files = [('resource', (filename, open(filename, 'rb'),"image/png"))]
      headers = {}
      response = requests.request(
        'POST', url, headers=headers, data=payload, files=files, verify=False)
    # print(response.text)
      if response.status_code == 200:
          print('Successfully sent to AutoAI', end="\r")
          return True
      else:
          print('Error while sending to AutoAI')
          return False
  except:
      print("failed")
def send_text_localization(image_path, bboxes, labels):
    rlef_format = rlh.segmentation_annotation_rlef(bboxes, labels)
    rlh.send_to_rlef(image_path, model_id = Config.TEXT_DETECTION_MODEL_ID , tag = "runtime-US", 
                     label = "runtime", status = "backlog", annotation=rlef_format)
    
    
     
