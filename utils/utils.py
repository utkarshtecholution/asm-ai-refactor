# import cv2
# import time
# import uuid
# import traceback

# def process_image_ocr(high_res_frame):
#     """
#     Processes an image frame using OCR (Optical Character Recognition) and associated image processing techniques.

#     This function performs several steps to extract text from an image frame, including image preprocessing, 
#     border removal, mask generation, angle correction, and OCR. It also measures and logs the time taken 
#     for each step. The function saves intermediate and final images for debugging and validation purposes, 
#     and updates a CSV file with timing data for performance profiling.

#     Args:
#         high_res_frame (numpy.ndarray): The high-resolution image frame to be processed. This should be 
#                                          a 3-dimensional numpy array (height x width x channels).

#     Returns:
#         dict or None: Returns a dictionary containing extracted SKU details if text is successfully detected. 
#                       Returns None if no text is detected or if an error occurs.

#     Raises:
#         Exception: Raises an exception if the resized mask and the original image do not have the same dimensions.

#     Side Effects:
#         - Saves images to disk for testing and logging purposes.
#         - Updates a CSV file (Config.TIME_PROFILE_PATH) with timing information for profiling the various stages of processing.
#         - Logs detailed timing information and error messages to the console.

#     Notes:
#         - Assumes the existence of external functions and modules such as `remove_boarder`, `dis_inference`, 
#           `angle_correction`, `get_best_rotated_image`, `mask_helper`, `TextDecoder`, and `image_utils`.
#         - Uses threading to save images asynchronously.
#         - Relies on `firebase_update` for status updates and `uuid` for generating unique filenames.
#     """
#     # df = pd.read_csv(Config.TIME_PROFILE_PATH)
#     # del df["Unnamed: 0"]
#     try:
        
#         firebase_update.update_status_workstation_ui(str(uuid.uuid1()))
#         text_decoder = TextDecoder(ocr_model_type="google-ocr")
#         s = time.time()
#         # print('>>>>> ONBOARDING OCR <<<<<<<<<')

#         ## Saving the Runtime image for testing purposes

#         threads_executor.submit(image_utils.save_image, f"runtimeLog/testing_images/{uuid.uuid1()}.png", high_res_frame)
        
#         image_path = f'runtimeLog/crop/{uuid.uuid1()}.png'

#         high_res_frame = remove_boarder.remove_border_pixels(high_res_frame, border_size=4)


#         dis_st = time.time()
#         mask = dis_inference.inference(cv2.resize(high_res_frame, Config.LOW_RES_SHAPE[::-1],  cv2.INTER_LINEAR))
#         dis_et = time.time()
        
#         print(f'Resizing image and dis inferencing takes : {dis_et-dis_st}')

#         resized_mask = cv2.resize(mask, high_res_frame.shape[:2][::-1], cv2.INTER_CUBIC)
        
#         if high_res_frame.shape[:2] != resized_mask.shape:
#             print(f'The resized mask shape is {resized_mask.shape}')
#             print(f'The image shape is {high_res_frame.shape}')
#             raise Exception("Both Image and Masks are not of same size, check line 32 in process.py")
        

#         # cv2.imwrite('runtimeLog/dis_mask.png', resized_mask) ## Just for logging purpose for validating DIS inference


#         threads_executor.submit(image_utils.save_image, 'runtimeLog/dis_mask.png', resized_mask) 




#         angle_st = time.time()
#         all_angles_list = angle_correction.all_angles(resized_mask, high_res_frame)
#         angle_et = time.time()
#         angle_prediction_time = angle_et - angle_st 

#         print(f'{angle_et - angle_st} for predicting the correct angle orientation')
       
        
        
#         best_st = time.time()
#         best_image, best_mask = get_best_rotated_image(high_res_frame, resized_mask, all_angles_list)
#         best_et = time.time()
#         angle_correction_time = best_et - best_st 
#         print(f'>>>>>>>>  {best_et - best_st} for getting the best frame <<<<<<<<<<')

        
#         cropped_image = mask_helper.cropped_image(best_image, best_mask) 
   
#         if cropped_image is not None:
#             text_decoder = TextDecoder(ocr_model_type="google-ocr")
        
#             threads_executor.submit(image_utils.save_image,image_path, cropped_image)
#             text_decoder.image_path = image_path
#             sku_details = text_decoder.decode_image_text(cropped_image)
#             if sku_details is None:
#                 print('NO TEXT DETECTIONS FOUND')
#                 # data = {'image_size' : [cropped_image.shape], 
#                 #     'angle_prediction' : [angle_prediction_time], 
#                 #     'angle_correction' :[angle_correction_time], 
#                 #     'dis_time' : [dis_et-dis_st], 
#                 #     'ocr_yolo' : [text_decoder.yolo_time], 
#                 #     'google_ocr' : [text_decoder.google_ocr_time], 
#                 #     'api_calls' : [text_decoder.api_calls],
#                 #     'Total Time' : [e-s]
#                 #     }
#                 # print(data)
#                 # df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

#                 # df.to_csv(Config.TIME_PROFILE_PATH)
#                 return None
#             e = time.time()
           
#             print(f'>>>>>>>>>> {e-s} for doing the whole OCR process <<<<<<<<<')
            
            
#             # data = {'image_size' : [cropped_image.shape], 
#             #         'angle_prediction' : [angle_prediction_time], 
#             #         'angle_correction' :[angle_correction_time], 
#             #         'dis_time' : [dis_et-dis_st], 
#             #         'ocr_yolo' : [text_decoder.yolo_time], 
#             #         'google_ocr' : [text_decoder.google_ocr_time], 
#             #         'api_calls' : [text_decoder.api_calls],
#             #         'Total Time' : [e-s]
#             #         }
#             # print(data)
#             # df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

#             # df.to_csv(Config.TIME_PROFILE_PATH)

#             return sku_details
        

#         else:
#             ### Fall back case if DIS fails to generate the segment
#             cv2.imwrite(image_path, high_res_frame)
#             text_decoder.image_path = image_path
#             sku_details = text_decoder.decode_image_text(image_path)
#             if sku_details is None:
#                 print('No text detections found in the captured image')
#                 return None 
            
#             e = time.time()
#             # data = {'image_size' : [cropped_image.shape], 
#             #         'angle_prediction' : [angle_prediction_time], 
#             #         'angle_correction' :[angle_correction_time], 
#             #         'dis_time' : [dis_et-dis_st], 
#             #         'ocr_yolo' : [text_decoder.yolo_time], 
#             #         'google_ocr' : [text_decoder.google_ocr_time], 
#             #         'api_calls' : [text_decoder.api_calls],
#             #         'Total Time' : [e-s]
#             #         }
#             # print(data)
#             # df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

#             # df.to_csv(Config.TIME_PROFILE_PATH)
#             print(f'>>>>>>>> {e-s} for inferencing one image in fallback condition <<<<<<<')
#             return sku_details
#     except Exception as e:
#         traceback.print_exc()
    