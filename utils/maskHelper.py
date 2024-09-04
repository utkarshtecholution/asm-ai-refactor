import cv2
import warnings
import numpy as np
import pandas as pd
# from rlef_utils import rlef_helper as rlh 


def generate_points_inside_polygon(polygon_coords, num_points):
    """
    Generate random points inside a polygon.

    Parameters:
        polygon_coords (list): Coordinates of the polygon vertices.
        num_points (int): Number of points to generate.

    Returns:
        list: Generated random points inside the polygon.
    """
    min_x, min_y = np.min(polygon_coords, axis=0)
    max_x, max_y = np.max(polygon_coords, axis=0)

    random_points = []

    while len(random_points) < num_points:
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)

        if is_point_inside_polygon(x, y, polygon_coords):
            random_points.append((x, y))

    return random_points

def is_point_inside_polygon(x, y, polygon_coords):
    """
    Check if a point is inside a polygon.

    Parameters:
        x (float): X-coordinate of the point.
        y (float): Y-coordinate of the point.
        polygon_coords (list): Coordinates of the polygon vertices.

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

class MaskImage:
    def __init__(self):
        self.image = None
        self.mask = None
        self.left = None
        self.right = None
        self.points = None
        self.annotations = None
        self.masked_image = None
        self.area = None
        self.temp = None
        self.new_image = None

    def calculate_area(self, shape):
        """
        Calculate the area of a polygon.

        Parameters:
            shape (list): List of (x, y) coordinates of the polygon vertices.

        Returns:
            float: Area of the polygon.
        """
        area = 0
        n = len(shape)
        for i in range(n):
            x1, y1 = shape[i]
            x2, y2 = shape[(i + 1) % n]
            area += (x1 * y2 - x2 * y1)
        return abs(area) / 2

    def maskProcessor(self, mask, tolerance=20, returnAll=False):
        """
        Process the mask to extract contours and calculate areas.

        Parameters:
            mask (np.ndarray): Input mask.
            tolerance (int): Tolerance for contour approximation.
            returnAll (bool): Whether to return all points.

        Returns:
            list, float: Contour points and the largest area value.
        """
        self.mask = mask

        if mask.any():
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            simplified_contours = []
            for contour in contours:
                simplified_contour = cv2.approxPolyDP(contour, tolerance, True)
                if simplified_contour.shape[0] > 1:
                    simplified_contour = simplified_contour.squeeze()
                    simplified_contours.append(simplified_contour.tolist())

            points = [shape for shape in simplified_contours if len(shape) >= 3]
            points = sorted(points, key=self.calculate_area, reverse=True)
            largest_area = max(points, key=self.calculate_area)
            largest_area_value = self.calculate_area(largest_area)
            
            self.area = largest_area_value
            points = [shape for shape in points if (self.calculate_area(shape) / largest_area_value) * 100 >= 0.75]

            self.points = points[0]
            return points, largest_area_value
        else:
            print('>>>>>>>>>>>>>>>>>> Mask Not Initiated <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            return None, None

    def find_contours(self, mask):
        """
        Find contours in a mask.

        Parameters:
            mask (np.ndarray): Input mask.

        Returns:
            int, float: Number of significant contours and the maximum area.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_contour_area = 1000
        significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_contour_area]
        num_significant_contours = 0
        max_area = 0
        largest_contour = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                largest_contour = contour
            if area > min_contour_area:
                num_significant_contours += 1
        print('CONTOUR MAX AREA', max_area)
        return num_significant_contours, max_area

    def resize_binary_mask(self, mask, size):
        """
        Resize a binary mask.

        Parameters:
            mask (np.ndarray): Input mask.
            size (tuple): New size (width, height).

        Returns:
            np.ndarray: Resized mask.
        """
        return cv2.resize(mask, size, cv2.INTER_LINEAR)

    def cropped_image(self, image_rgb, mask, annotations=None, padding=15, point_sensitivity = 20):
        """
        Create a masked image based on the mask and annotations.

        Parameters:
            image_rgb (np.ndarray): Input image in RGB format.
            mask (np.ndarray): Binary mask.
            annotations (list): Annotations for the bounding box.
            padding (int): Padding for the bounding box.

        Returns:
            np.ndarray: Masked image.
        """
        try:
            if annotations is None:
                points, area = self.maskProcessor(mask, point_sensitivity)
                print(f'>>>>>>>>>>>> Area Value is {area} <<<<<<<<<<<<<<<<<<<<<<<<<<')
                if area < 10000:
                    return None 
                self.points = points 
                xyxy = self.find_left_upper_right_down(points[0])
                print('######### ANNOTATIONS ARE : ')
                print(xyxy)
            annotations = xyxy
            # mask = self.resize_binary_mask(mask, (image_rgb.shape[1], image_rgb.shape[0]))
            xmin, ymin, xmax, ymax = annotations[0], annotations[1], annotations[2], annotations[3]
            # inv_final_mask = ~mask
            # test_image = image_rgb.copy()
            # test_image_arr = np.array(test_image)
            # masked_out_image = np.copy(test_image_arr)
            # masked_out_image[inv_final_mask] = [0, 0, 0]
            # print('IMAGE CROPPED')
            # # try:
            # #     masked_out_image = masked_out_image[ymin - padding:ymax + padding, xmin - padding:xmax + padding]
            # # except Exception as e:
            image_rgb = image_rgb[ymin:ymax, xmin:xmax]
        
            return image_rgb
        except Exception as e:
            print(f'{e} exception at line 205 in maskHelper.py ......')
            return None

    def find_left_upper_right_down(self, points):
        """
        Find the left-upper and right-down corners of a bounding box.

        Parameters:
            points (list): List of (x, y) coordinates of the polygon vertices.

        Returns:
            list: Coordinates of the bounding box [xmin, ymin, xmax, ymax].
        """
        if not points:
            warnings.warn('>>> WARNING : list of points is empty. Please Check <<<')
            return None, None
        
        left_upper = [min(points, key=lambda point: point[0])[0], min(points, key=lambda point: point[1])[1]]
        right_down = [max(points, key=lambda point: point[0])[0], max(points, key=lambda point: point[1])[1]]
        annotations = [left_upper[0], left_upper[1], right_down[0], right_down[1]]
        return annotations

 

    def get_xyxy_mask(self, mask):
        """
        Get bounding box coordinates from the mask.

        Parameters:
            mask (np.ndarray): Binary mask.

        Returns:
            list: Bounding box coordinates.
        """
        try:
            points, _ = self.maskProcessor(mask, 20)
        except Exception as e:
            print(e)
            return None
        bounding_boxes = self.get_xyxy(points)
        return bounding_boxes

    def get_xyxy(self, points_list):
        """
        Get bounding box coordinates from the list of points.

        Parameters:
            points_list (list): List of points.

        Returns:
            list: Bounding box coordinates.
        """
        if points_list is None:
            vertices_list = self.points
        else:
            vertices_list = points_list
        
        bounding_boxes = []
        for entry in vertices_list:
            bounding_boxes.append(self.find_left_upper_right_down(entry))

        return bounding_boxes

    def get_new_object(self, prev_mask, current_mask):
        """
        Get the new object by subtracting the previous mask from the current mask.

        Parameters:
            prev_mask (np.ndarray): Previous mask.
            current_mask (np.ndarray): Current mask.

        Returns:
            np.ndarray: Mask of the new object.
        """
        current_contour, current_area = self.find_contours(current_mask)
        previous_contour, previous_area = self.find_contours(prev_mask)

        if previous_contour < current_contour:
            print('####### FIRST CONDITION IF THERE IS NO OVERLAP ')
            return cv2.subtract(current_mask, prev_mask)

        overlap_mask = cv2.bitwise_and(prev_mask, current_mask)
        combined_mask = cv2.bitwise_or(overlap_mask, current_mask)
        return combined_mask

    def maskSubtract(self, prevMask, currentMask):
        """
        Subtract the previous mask from the current mask.

        Parameters:
            prevMask (np.ndarray): Previous mask.
            currentMask (np.ndarray): Current mask.

        Returns:
            np.ndarray: Resulting mask.
        """
        return cv2.subtract(currentMask, prevMask)

    def generateMaskFromPolygon(self, polygon, image_shape=None):
        """
        Generate a mask from polygon coordinates.

        Parameters:
            polygon (np.ndarray): Polygon coordinates.
            image_shape (tuple): Shape of the image (height, width).

        Returns:
            np.ndarray: Generated mask.
        """
        if image_shape is not None:
            mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        return mask

    def generatePointsFromSegment(self, mask):
        """
        Generate points from the mask segmentation.

        Parameters:
            mask (np.ndarray): Binary mask.

        Returns:
            list: Points from the segmentation.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        l = []  # list to store each contour

        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            l.append([i, area, c])

        df = pd.DataFrame(l, columns=['index', 'Area', "contour"])
        df = df.sort_values("Area", ascending=False)
        df.reset_index(inplace=True)
        n = df["index"][0]
        c = df["contour"][0]
        points = self.contour_to_segmentation_points(c)
        return points

    def contour_to_segmentation_points(self, contour):
        """
        Convert a single OpenCV contour to segmentation points in (x, y) format.

        Parameters:
            contour (np.ndarray): Contour points from OpenCV.

        Returns:
            list: Segmentation points in (x, y) format.
        """
        segmentation_points = []
        for point in contour:
            x, y = point[0]
            segmentation_points.append((x, y))
        return segmentation_points