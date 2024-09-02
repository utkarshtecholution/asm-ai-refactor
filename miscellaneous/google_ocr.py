import io
import os
from google.cloud import vision


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "firebase_keys/proj-qsight-asmlab-b5530bda7bad.json"

client = vision.ImageAnnotatorClient()

# def initialize_vision_client(credentials_path):
#     """
#     Initialize the Google Cloud Vision API client.

#     Parameters:
#         credentials_path (str): Path to the Google Cloud service account key file.

#     Returns:
#         vision.ImageAnnotatorClient: The Vision API client.
#     """
#     os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
#     return vision.ImageAnnotatorClient()

def load_image(path):
    """
    Load an image from the specified path.

    Parameters:
        path (str): Path to the image file.

    Returns:
        bytes: The image content in bytes.
    """
    with io.open(path, 'rb') as image_file:
        return image_file.read()

def perform_text_detection(client, image_content):
    """
    Perform document text detection on the given image content using Google Cloud Vision API.

    Parameters:
        client (vision.ImageAnnotatorClient): The Vision API client.
        image_content (bytes): The image content in bytes.

    Returns:
        list: List of detected text, bounding boxes, and confidence scores.
    """
    image = vision.Image(content=image_content)
    response = client.document_text_detection(image=image)
    result_list = []

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                bbox = paragraph.bounding_box
                text = ''
                confidence = 1

                for word in paragraph.words:
                    for symbol in word.symbols:
                        text += symbol.text
                        if symbol.confidence < confidence:
                            confidence = symbol.confidence

                result_list.append({
                    "text": text,
                    "bbox": {
                        "vertices": [
                            {"x": vertex.x, "y": vertex.y} for vertex in bbox.vertices
                        ]
                    },
                    "confidence": confidence
                })

    return result_list

def cloud_vision_inference(image_path, queue_obj = None ):
    """
    Perform document text detection on an image using Google Cloud Vision API.

    Parameters:
        image_path (str): Path to the image file.
        credentials_path (str): Path to the Google Cloud service account key file.

    Returns:
        list: List of detected text, bounding boxes, and confidence scores.
    """
    # client = initialize_vision_client(credentials_path)
    image_content = load_image(image_path)
    text_results = perform_text_detection(client, image_content)
    if queue_obj is not None:
        queue_obj.put(text_results)
    return text_results
