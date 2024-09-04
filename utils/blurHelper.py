import cv2 


class BlurFilter:
    def __init__(self, threshold = 80):
        self.blur_threshold = threshold 

    def detect_blur(self, frame):
        # Your logic to determine the best frame using detect_blur_fft
        current_score = self.detect_blur_laplacian(frame)
        print(current_score)
        # If current score is greater than threshold, then we say the frame is blur else it's not blur 
        return current_score < self.blur_threshold
    
    def detect_blur_laplacian(self, img_arr):
        gray_frame = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray_frame, cv2.CV_64F).var()

        return variance




        
        
