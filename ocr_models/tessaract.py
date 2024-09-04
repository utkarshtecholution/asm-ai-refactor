import tesserocr as tocr
from PIL import Image

def tesseract_ocr_inference(image_path):
    """
    Perform OCR on the given image using Tesseract-OCR.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        str: Extracted text from the image.
    """
    try:
        image = Image.open(image_path)
        results = tocr.image_to_text(image)
        print(results)
        return results
    except Exception as e:
        print(f"An error occurred during OCR processing: {e}")
        return None
