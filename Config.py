STREAMING_FRAME_SIZE = (640, 480)
import cv2
from collections import deque
import numpy as np
import os 
from concurrent.futures import ThreadPoolExecutor
from utils import maskHelper, polygonHelper, planeCorrection, blurHelper
# from ocr_utils.textDecoder import TextDecoder
import os 
from threading import Lock 


HEADERS = {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcHBOYW1lIjoiUHJvZHVjdCBPbmJvYXJkaW5nIiwidXNlckVtYWlsIjoiZGl2eWFuc2gua3VtYXJAdGVjaG9sdXRpb24uY29tIiwidXNlcklkIjoiNjQ5MThiNTE2NDkzYTk2NTI5ODM3MzgwIiwic2NvcGVPZlRva2VuIjp7InByb2plY3RJZCI6IjY1MGFmNzk1ZWJlNzBmMWY0Zjc5YmYxZSIsInNjb3BlcyI6eyJwcm9qZWN0Ijp7InJlYWQiOnRydWUsInVwZGF0ZSI6dHJ1ZSwiZGVsZXRlIjp0cnVlLCJjcmVhdGUiOnRydWV9LCJyZXNvdXJjZSI6eyJyZWFkIjp0cnVlLCJ1cGRhdGUiOnRydWUsImRlbGV0ZSI6dHJ1ZSwiY3JlYXRlIjp0cnVlfSwibW9kZWwiOnsicmVhZCI6dHJ1ZSwidXBkYXRlIjp0cnVlLCJkZWxldGUiOnRydWUsImNyZWF0ZSI6dHJ1ZX0sImRhdGFTZXRDb2xsZWN0aW9uIjp7InJlYWQiOnRydWUsInVwZGF0ZSI6dHJ1ZSwiZGVsZXRlIjp0cnVlLCJjcmVhdGUiOnRydWV9LCJtb2RlbENvbGxlY3Rpb24iOnsicmVhZCI6dHJ1ZSwidXBkYXRlIjp0cnVlLCJkZWxldGUiOnRydWUsImNyZWF0ZSI6dHJ1ZX0sInRlc3RDb2xsZWN0aW9uIjp7InJlYWQiOnRydWUsInVwZGF0ZSI6dHJ1ZSwiZGVsZXRlIjp0cnVlLCJjcmVhdGUiOnRydWV9LCJjb3BpbG90Ijp7InJlYWQiOnRydWUsInVwZGF0ZSI6dHJ1ZSwiZGVsZXRlIjp0cnVlLCJjcmVhdGUiOnRydWV9LCJjb3BpbG90UmVzb3VyY2UiOnsicmVhZCI6dHJ1ZSwidXBkYXRlIjp0cnVlLCJkZWxldGUiOnRydWUsImNyZWF0ZSI6dHJ1ZX0sIm1vZGVsR3JvdXAiOnsicmVhZCI6dHJ1ZSwidXBkYXRlIjp0cnVlLCJkZWxldGUiOnRydWUsImNyZWF0ZSI6dHJ1ZX19fSwiaWF0IjoxNzExNjQxMzMyfQ.D88X0Aix3IlDTWduT0Yq_BSHTHvGPczuUm_iKC0ygKs'
}

BASE_DIR = os.getcwd()
log_dir = os.path.join(BASE_DIR, "runtimeLog")



threads_executor = ThreadPoolExecutor(max_workers = 30)



# textDecoder = TextDecoder(ocr_model_type="google-ocr")
polyHelper = polygonHelper.PolygonHelper()
maskClass = maskHelper.MaskImage()
blur_helper = blurHelper.BlurFilter()





# Define constants and configuration parameters


MAIN_FRAME_QUEUE = deque(maxlen=500)
FONT = cv2.FONT_HERSHEY_SIMPLEX
ORG = (50, 50)
FONT_SCALE = 1
COLOR = (255, 125, 100)
THICKNESS = 2
QUEUE_ADD_INTERVAL = 3
CALIBRATION = True
CALIBRATION_TIME = 10  # seconds
FPS = 30
CALIBRATION_FRAME_COUNT = CALIBRATION_TIME * FPS
NUM_POINTS = 30
SIM_THRESH = 80  # Similarity threshold for motion detection
RACK_REFERENCE = "61c03eef-840c-49df-ba20-0952553a4953"
MOTION_THRESH = 0.85
EMPTY_FILL = False
LOW_RES_SHAPE = (480, 640)
HIGH_RES_SHAPE = (2160, 3840)
AF_LOW_RES = (720, 1280)
PLANOGRAM_REF_NO = ""
CURRENT_REF_NO = ""
BLUR_THRESHOLD = 900
RECORD_FLAG = False 


SMALLEST_AREA_VALUE = 10000

MIN_FOCUS_VALUE = 60
MAX_FOCUS_VALUE = 100
YOLO_THRESHOLD = 0.3


GCP_CREDS_PATH = "firebase_keys/proj-qsight-asmlab-b5530bda7bad.json"
REF_IMG_PATH = "runtimeLog/reference_image_2.jpg"
RUNTIME_IMG_PATH = "runtimeLog/action_image.jpg"
HIGH_RES_PATH = "runtimeLog/high_res_image.jpg"
OCR_CROP_IMG_PATH = "runtimeLog/ocr"
REF_MASK_PATH = "runtimeLog/ref_mask.jpg"
TEXT_LOCALIZATION_YOLO_PATH = "weights/text_localization_obb.pt"
DIS_MODEL_PATH = "workstation_model.pth"

CURRENT_FRAME_PATH = "runtimeLog/current_frame.jpg"
YOLO_ANNOT_IMG_PATH = "runtimeLog/annotated_image.jpg"
VIDEOS_DIR_PATH = "runtimeLog/videos"
SEGMENT_PATH = "runtimeLog/segment.jpg"
TIME_PROFILE_PATH = "runtimeLog/time_profiling.csv"

OCR_PROCESS_LOCK = Lock() 


# RLEF MODEL IDS 
DIS_MODEL_ID = "6685900f97eae5e91291d0f4"
OCR_MODEL_ID = "66a22683cc437fd26420d927"
GRABNGO_MODEL_ID = "66858feb97eae5e91291c33d"
TEXT_DETECTION_MODEL_ID = "66965ed548169ca1bf32ef13"
# Dummy database entry for tracking item information
DUMMY_DB = {
    'item_count' : "loading", 
    'Qsight_image' : "loading", 
    'assigned_compartment' : "loading", 
    'ref_no' : "loading", 
    'lot_no' : "loading", 
    'product_name' : "loading", 
    'use_by' : "loading",
    'barcode' : "loading",
    'captured_barcode' : "loading", 
    'captured_use_by' : "loading", 
    'captured_lot_no' : "loading", 
    'captured_ref_no' : "loading", 
    'captured_product_name': "loading",
    'product_name' : "loading"
}

# Example of a smaller open rack setup
# ROI_POLYGON = np.array([(179, 293), (389, 299), (460, 479), (149, 478), (80, 446)], dtype=np.int32)
# FOV_POLYGON = np.array([(74, 428), (3, 79), (454, 89), (454, 473)], dtype=np.int32)

## US RACK SETUP 
FOV_POLYGON = np.array([(353,7), (372, 477), (0, 474), (6,6)], dtype=np.int32)
ROI_POLYGON = np.array([(353,7), (372, 477), (0, 474), (6,6)], dtype=np.int32)

## US 
ONBOARD_POLYGON = np.array([(403, 28), (396, 695), (28, 705), (0,0)], dtype=np.int32)
ONBOARD_ROI = [ 0, 0, 396, 695]

# India 
# ONBOARD_POLYGON = np.array([(172,51), (161, 449), (467, 479), (464, 56)], dtype = np.int32)
ONBOARD_MASK = maskClass.generateMaskFromPolygon(ONBOARD_POLYGON, AF_LOW_RES )
ONBOARD_MASK_LOW_RES = cv2.resize(ONBOARD_MASK, LOW_RES_SHAPE[::-1], cv2.INTER_LINEAR)
cv2.imwrite('runtimeLog/onboard_mask.jpg', ONBOARD_MASK)

ONBOARD_REFERENCE = "ff0aa8e3-5584-4776-8024-994161d14433"

ROI_MASK = maskClass.generateMaskFromPolygon(ROI_POLYGON,image_shape=LOW_RES_SHAPE)
FOV_MASK = maskClass.generateMaskFromPolygon(FOV_POLYGON, image_shape = LOW_RES_SHAPE)
cv2.imwrite('runtimeLog/fov_mask.jpg', FOV_MASK)
cv2.imwrite('runtimeLog/roi_mask.jpg', ROI_MASK)


TOPIC_ID = "product-onboarding-workstation"
PROJECT_ID = "proj-qsight-asmlab"
SUBSCRIPTION_ID = "product-onboarding-workstation-sub"

US_TAG = "Live-data-US"
def initialize_system():
    """
    Initializes the motion detection and tracking system with predefined configurations.
    """
    print("Initializing system with the following configurations:")
    print(f"FPS: {FPS}")
    print(f"Calibration Time: {CALIBRATION_TIME} seconds")
    print(f"Similarity Threshold: {SIM_THRESH}")
    print(f"Motion Threshold: {MOTION_THRESH}")

if __name__ == "__main__":
    initialize_system()
    # Additional setup and motion detection logic can be implemented here