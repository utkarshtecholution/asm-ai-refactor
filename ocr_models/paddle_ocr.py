# import time
# from PIL import Image
# from paddleocr import PaddleOCR, draw_ocr

# class Pocr:
#     def __init__(self):
#         """
#         Initialize the PaddleOCR model.
#         """
#         self.model = PaddleOCR(use_angle_cls=True, lang='en')

#     def ocr(self, img):
#         """
#         Perform OCR on the given image.

#         Parameters:
#             img (str or np.ndarray): Path to the image file or image array.

#         Returns:
#             list: OCR results.
#         """
#         return self.model.ocr(img, cls=True)[0]

# def pocr_inference(image_path):
#     """
#     Perform OCR inference on the given image using PaddleOCR.

#     Parameters:
#         image_path (str): Path to the image file.

#     Returns:
#         list: Extracted text from the image.
#     """
#     print("="*100)
#     print("PaddleOCR Results")
#     start_time = time.time()
#     result = OCR.ocr(image_path, cls=True)
#     end_time = time.time()
#     print(f"Time taken: {end_time - start_time} seconds")

#     text = []
#     if result:
#         for idx in range(len(result)):
#             res = result[idx]
#             for line in res:
#                 print(line)
#                 text.append(line[1][0])
#     else:
#         print("No OCR results found.")

#     print("="*100)
#     return text

# def pocr_inference_2(image_path):
#     """
#     Perform OCR inference on the given image using PaddleOCR and draw the results.

#     Parameters:
#         image_path (str): Path to the image file.
#     """
#     OCR = PaddleOCR(use_angle_cls=True, lang='en')
#     boxes = []
#     txts = []
#     scores = []

#     print("="*100)
#     print("PaddleOCR Results")
#     result = OCR.ocr(image_path, cls=True)
    
#     for line in result:
#         boxes.append(line[0])
#         txts.append(line[1][0])
#         scores.append(line[1][1])
#         print(line)

#     # Draw result
#     print("Boxes:\n", boxes)
#     print("-"*50)
#     print("Texts:\n", txts)
#     print("-"*50)
#     print("Scores:\n", scores)

#     image = Image.open(image_path).convert('RGB')
#     im_show = draw_ocr(image, boxes, txts, scores, font_path='simfang.ttf')
#     im_show = Image.fromarray(im_show)
#     im_show.save('result.jpg')
#     print("Image Saved")

# OCR = PaddleOCR(use_angle_cls=True, lang='en')


from paddleocr import PaddleOCR, draw_ocr



class PaddleOCRInference:
    def __init__(self, custom_model_path = None):
        self.ocr_model = PaddleOCR(use_angle_cls=True, lang='en')
        
    def ocr_inference(self, image_path, return_text = True):
        result = self.ocr_model.ocr(image_path, cls = True)
        boxes, txts, scores = [], [], []
        print(result)
        if result[0] is None:
            return None 
        raw_text = ''

        for line in result:
            
            boxes.append(line[0][0])
            txts.append(line[0][1][0])
            raw_text +=  " " + str(line[0][1][0])
            scores.append(line[0][1][1])
        if return_text:
            return raw_text
        return boxes, txts, scores 
    

        

        


