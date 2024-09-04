import cv2
from skimage.metrics import structural_similarity as ssim
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import sys

# Append the current script directory to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)


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
        self.threshold = 0.99
        self.buffer = buffer
        self.count = 0
        self.current_motion_status = False 
        self.motion_check_count = 0

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
            if motion_status == True:
                self.motion_check_count+=1 
                # print(self.motion_check_count)
                if self.motion_check_count > 3:
                    self.motion = True 
            
            else:
                self.motion = False 
                self.motion_check_count = 0 
            # print(ssim_index)
            print(f'Final Motion Status: {self.motion}')
                
        except Exception as e:
            logging.error(f"Motion detection failed due to: {e}")
            raise e

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
