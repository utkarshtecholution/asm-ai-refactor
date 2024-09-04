from collections import deque
import threading
from ai_pipeline.motion import MotionDetection
from ai_pipeline.angleCorrection import process_image_ocr
import cv2

# creating an frame queue for inserting realtime frames
frame_queue = deque(maxlen=5)

new_item_event = threading.Event()

def consumer():
    # Motion Detection
    motion_detector = MotionDetection()
    skip_frame = 3
    current_frame_count = 0
    while True:
        new_item_event.wait()  # Wait until the event is set
        while frame_queue:
            frame = frame_queue.popleft()
            current_frame_count += 1

            # Feeding every 3rd frame for checking motion dectection 
            if current_frame_count%skip_frame==0:
                resized_frame = cv2.resize(frame, (640, 480), cv2.INTER_CUBIC)
                motion_detector.motionUpdate(resized_frame)

            if motion_detector.prev_motion_status == True and motion_detector.current_motion_status == False:
                print("Motion stopped")
                # TODO: Need to have a 3-5 frame buffer for finding best frame
                process_image_ocr(frame)

            print("consumed...")
        new_item_event.clear()  # Clear the event when queue is empty

consumer_thread = threading.Thread(target=consumer, daemon=True)
consumer_thread.start()
