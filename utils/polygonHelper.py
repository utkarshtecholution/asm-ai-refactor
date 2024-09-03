import cv2
import numpy as np
from utils.maskHelper import MaskImage

maskHelper = MaskImage()

class PolygonHelper:
    def __init__(self):
        self.polygon = None

    def generate_points_inside_polygon(self, polygon_coords, num_points):
        """
        Generate random points inside a polygon.

        Parameters:
            polygon_coords (np.ndarray): Coordinates of the polygon vertices.
            num_points (int): Number of points to generate inside the polygon.

        Returns:
            list: Generated random points inside the polygon.
        """
        min_x, min_y = np.min(polygon_coords, axis=0)
        max_x, max_y = np.max(polygon_coords, axis=0)

        random_points = []

        while len(random_points) < num_points:
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)

            if self.is_point_inside_polygon(x, y, polygon_coords):
                random_points.append((x, y))

        return random_points

    def is_point_inside_polygon(self, x, y, polygon_coords):
        """
        Check if a point is inside a polygon.

        Parameters:
            x (float): X-coordinate of the point.
            y (float): Y-coordinate of the point.
            polygon_coords (np.ndarray): Coordinates of the polygon vertices.

        Returns:
            bool: True if the point is inside the polygon, False otherwise.
        """
        n = len(polygon_coords)
        inside = False

        p1x, p1y = polygon_coords[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon_coords[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def check_seg_in_roi(self, mask, roi, point_thresh=2):
        """
        Check if a segmentation mask is within the region of interest (ROI).

        Parameters:
            mask (np.ndarray): Binary mask.
            roi (np.ndarray): Coordinates of the ROI polygon vertices.
            point_thresh (int): Threshold for the number of points inside the ROI to consider the mask within the ROI.

        Returns:
            bool: True if the mask is within the ROI, False otherwise.
        """
        try:
            polygons, _ = maskHelper.maskProcessor(mask, 1)
        except Exception as e:
            print(e)
            return False

        points = self.generate_points_inside_polygon(polygons[0], 30)
        num_points = sum(1 for point in points if self.is_point_inside_polygon(point[0], point[1], roi))

        return num_points > point_thresh
