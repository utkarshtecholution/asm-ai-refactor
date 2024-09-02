import cv2
import time
import os
import numpy as np
from math import atan2, cos, sin, sqrt, pi
import traceback 

class ImageOrientationCorrection:
    def __init__(self, angle=0):
        self.angle = angle

    def draw_axis(self, img, start_point, end_point, color, thickness):
        """
        Draws an axis/line on the image from start_point to end_point with a specified color and thickness.
        :param img: Image to draw the axis on.
        :param start_point: Starting point of the axis (x, y).
        :param end_point: Ending point of the axis (x, y).
        :param color: Color of the axis (B, G, R).
        :param thickness: Thickness of the axis line.
        """
        angle = atan2(start_point[1] - end_point[1], start_point[0] - end_point[0])
        hypotenuse = sqrt((start_point[1] - end_point[1]) ** 2 + (start_point[0] - end_point[0]) ** 2)
        end_point = (start_point[0] - thickness * hypotenuse * cos(angle),
                     start_point[1] - thickness * hypotenuse * sin(angle))

        cv2.line(img, (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1])), color, 3,
                cv2.LINE_AA)

        # Create the arrow hooks
        for sign in [1, -1]:
            hook_point = (end_point[0] + 9 * cos(angle + sign * pi / 4),
                          end_point[1] + 9 * sin(angle + sign * pi / 4))
            cv2.line(img, (int(hook_point[0]), int(hook_point[1])), (int(end_point[0]), int(end_point[1])), color, 3,
                    cv2.LINE_AA)

    def get_orientation(self, pts, img):
        """
        Determines the orientation of the contour specified by pts in the given image.
        :param pts: Contour points to find the orientation of.
        :param img: Image containing the contour.
        :return: Angle of orientation in degrees.
        """
        sz = len(pts)
        data_pts = np.float32(pts).reshape(sz, 2)

        # Perform PCA analysis
        mean, eigenvectors = cv2.PCACompute(data_pts, mean=np.array([]))

        # Store the center of the object
        m = cv2.moments(pts)
        if m['m00'] != 0:
            cntr = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
        else:
            cntr = (int(mean[0, 0]), int(mean[0, 1]))

            # Draw the principal components
        cv2.circle(img, cntr, 3, (255, 0, 255), 2)
        p1 = (cntr[0] + 0.02 * eigenvectors[0, 0] * 30,
              cntr[1] + 0.02 * eigenvectors[0, 1] * 30)  # Scale factor for visualization
        p2 = (cntr[0] - 0.02 * eigenvectors[1, 0] * 30,
              cntr[1] - 0.02 * eigenvectors[1, 1] * 30)

        self.draw_axis(img, cntr, p1, (0, 255, 0), 1)
        self.draw_axis(img, cntr, p2, (255, 255, 0), 5)

        angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # Orientation in radians
        angle = -np.rad2deg(angle)  # Convert to degrees
        return angle

    def measure_angle(self, src, points):
        """
        Measures the orientation angle of a contour specified by points in the source image.
        :param src: Source image containing the contour.
        :param points: Contour points to measure the angle of.
        :return: Tuple of the measured angle and its opposite angle.
        """
        if src is None:
            return 0, 0
        start_time = time.time()
        angle = round(self.get_orientation(points, src.copy()), 2)
        # print(f" angle prediction from opencv took {time.time() - start_time} seconds")
        return angle, angle + 180


def save_image(path, image):
    """
    Saves an image to the specified directory. If the directory does not exist, it creates it.
    
    Parameters:
    path (str): The file path where the image will be saved.
    image (numpy.ndarray): The image to be saved.
    
    Returns:
    None
    """
    try:


        st = time.time()
        
        # Save the image to the specified path
        cv2.imwrite(path, image)
        
        # Print the time taken to save the image
        print(f"Image saving took {time.time() - st} seconds")
    except Exception as e:
        traceback.print_exc()
