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
    def __init__(self, queue_len=10, max_workers=1, buffer = 3):
        """
        Initialize the MotionDetection class.
        
        Parameters:
            queue_len (int): Maximum length of the frame queue.
            max_workers (int): Maximum number of worker threads.
        """
        self.motionValue = 1000
        self.frameQueue = deque(maxlen=queue_len)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.threshold = 0.95
        self.buffer = buffer
        self.count = 0
        self.prev_motion_status = False
        self.current_motion_status = False
        self.motion_check_count = 0

    def motion_detect(self, frame1, frame2, log=False):
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
                print(f"Similarity Index: {ssim_index}")
                # logging.info(f"Similarity Index: {ssim_index}")

            motion_status = ssim_index < self.threshold

            if motion_status == True:
                self.motion_check_count+=1 
                # print(self.motion_check_count)
                if self.motion_check_count > 3:
                    self.current_motion_status = True 
            
            else:
                self.current_motion_status = False 
                self.motion_check_count = 0 
            # print(ssim_index)
            # print(f'Final Motion Status: {self.motion}')
                
        except Exception as e:
            logging.error(f"Motion detection failed due to: {e}")
            raise e

    def motionUpdate(self, frame):
        """
        Update the motion detection with a new frame.
        
        Parameters:
            frame (np.ndarray): New frame to update motion detection.
        """
        self.prev_motion_status = self.current_motion_status
        self.frameQueue.append(frame)
        if len(self.frameQueue) > 1:
            self.motion_detect(self.frameQueue.popleft(), self.frameQueue.popleft(), False)
            # self.executor.submit(self.motion_detect, self.frameQueue.popleft(), self.frameQueue.popleft())
