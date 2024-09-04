import cv2
from skimage.metrics import structural_similarity as ssim
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import sys
from Config import MOTION_THRESH
from utils.polygonHelper import PolygonHelper

# Append the current script directory to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

LOG_FOLDER_PATH = 'runtimeLog/dynamicROI'
polyHelper = PolygonHelper()

class MotionDetection:
    def __init__(self, queue_len=10, max_workers=5, buffer = 3):
        """
        Initialize the MotionDetection class.
        
        Parameters:
            queue_len (int): Maximum length of the frame queue.
            max_workers (int): Maximum number of worker threads.
        """
        self.motionValue = 1000
        self.motion = False
        self.frameQueue = deque(maxlen=queue_len)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.threshold = MOTION_THRESH
        self.buffer = buffer
        self.count = 0

    def motionDetect(self, frame1, frame2, log=False):
        """
        Detect motion by comparing two frames using Structural Similarity Index (SSIM).
        
        Parameters:
            frame1 (np.ndarray): First frame.
            frame2 (np.ndarray): Second frame.
            log (bool): Whether to log the SSIM index.
        """
        try:
            gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            ssim_index, _ = ssim(gray_frame1, gray_frame2, full=True)
            self.motionValue = ssim_index
            if log:
                logging.info(f"Similarity Index: {ssim_index}")

            motion_status = ssim_index < self.threshold

            if motion_status is True: 
                self.count +=1 
            else:
                self.count = 0

            if self.count != 0 and self.count % self.buffer ==0:
                self.motion = True 

            else:
                self.motion = False 
                
        except Exception as e:
            logging.error(f"Motion detection failed due to: {e}")
            raise e

    def motionUpdate(self, frame):
        """
        Update the motion detection with a new frame.
        
        Parameters:
            frame (np.ndarray): New frame to update motion detection.
        """
        self.frameQueue.append(frame)
        if len(self.frameQueue) > 1:
            self.executor.submit(self.motionDetect, self.frameQueue[-1], self.frameQueue[-2])

    def motionCalibration(self, frame):
        """
        Calibrate the motion detection threshold using a new frame.
        
        Parameters:
            frame (np.ndarray): New frame for calibration.
        """
        print("########### Don't perform any motion in the rack ###################")
        self.motionUpdate(frame)
        if self.motionValue < self.threshold:
            self.threshold = self.motionValue

    def updateThreshold(self, new_threshold):
        """
        Update the motion detection threshold.
        
        Parameters:
            new_threshold (float): New threshold value.
        """
        self.threshold = new_threshold

    def getMotionROI(self, frame, mask, default_mask, polygon):
        """
        Get the Region of Interest (ROI) for motion detection.
        
        Parameters:
            frame (np.ndarray): Current frame.
            mask (np.ndarray): Mask for the current frame.
            default_mask (np.ndarray): Default mask.
            polygon (list): List of polygon coordinates.
        
        Returns:
            tuple: ROI frame and current mask.
        """
        if polyHelper.check_seg_in_roi(mask, polygon):
            binary_mask = cv2.bitwise_or(default_mask, mask)
            cv2.imwrite(f'{LOG_FOLDER_PATH}/joined_mask.jpg', binary_mask)
            roi_frame = cv2.bitwise_and(frame, frame, mask=binary_mask)
            current_mask = binary_mask
            cv2.imwrite(f'{LOG_FOLDER_PATH}/initial_mask.jpg', current_mask)
        else:
            roi_frame = cv2.bitwise_and(frame, frame, mask=default_mask)
            current_mask = default_mask

        return roi_frame, current_mask
