from collections import deque
import threading
import cv2
from ai_pipeline.motion import MotionDetection
from ai_pipeline.main import main

# creating an frame queue for inserting realtime frames
frame_queue = deque(maxlen=5)

new_item_event = threading.Event()

def consumer(log=False):
    # Motion Detection
    motion_detector = MotionDetection()
    skip_frame = 3
    current_frame_count = 0
    while True:
        new_item_event.wait()  # Wait until the event is set
        while frame_queue:
            frame, low_res_frame = frame_queue.popleft()
            current_frame_count += 1

            ## Motion Dectection
            # Feeding every 3rd frame for checking motion dectection 
            if current_frame_count%skip_frame==0:
                motion_detector.motionUpdate(low_res_frame)
            
            ## Object Detection and Angle Correction
            if motion_detector.prev_motion_status == True and motion_detector.current_motion_status == False:
                if log:
                    print("Motion stopped")
                main(frame, low_res_frame)

            if log:
                print("consumed...")
        new_item_event.clear()  # Clear the event when queue is empty

consumer_thread = threading.Thread(target=consumer, daemon=True)
consumer_thread.start()
