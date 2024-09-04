from ai_pipeline.angleCorrection import process_image_ocr
from ai_pipeline.DISegmention import dis_object_seg
from ai_pipeline.angleCorrection import angle_correction
from ai_pipeline.image_crop import crop_oriented_bbox_image
from ai_pipeline.OCR_inference import text_ocr_process_single_image, remove_unwanted_text

def main(high_res_frame, low_res_frame):

    # Getting mask of segments of Object using DIS Model
    low_res_mask = dis_object_seg(low_res_frame)
    if low_res_mask is None:
        return
    
    # Angle Prediction from the mask and returns label detection results
    results = angle_correction(high_res_frame, low_res_mask)

    # Croping images
    final_result = {}
    for segment in results:
        crop_img_path = crop_oriented_bbox_image(image = high_res_frame, bbox = segment['xyxy'], angle = None,  
                                                            oriented_box= segment['obb'], save_image= True)
        
        ocr_text = text_ocr_process_single_image(crop_img_path, segment['clss'])
        ocr_text = remove_unwanted_text(segment['clss'], ocr_text)

        if segment['clss'] in final_result:
            if len(final_result['clss'])>len(ocr_text):
                final_result['clss']['text'] = ocr_text
                final_result['clss']['conf'] = segment["conf"]

        else:
            final_result[segment['clss']] = {
                "type": segment['clss'],
                "text" : ocr_text,
                "conf" : segment["conf"]
            }

    return final_result    
        
if __name__=="__main__":
    main()