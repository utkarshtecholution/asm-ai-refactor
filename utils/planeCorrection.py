import math
import cv2
import numpy as np
import sympy
import concurrent.futures
from utils.image_utils import save_image
import time
import os
from rlef_utils import send_to_rlef


def reorder_points(points, clockwise=True):

    """

    Reorder a list of points to be sorted in a clockwise or counterclockwise manner around their centroid.



    :param points: A list of (x, y) tuples representing the points to be reordered.

    :param clockwise: A boolean indicating the desired order. If True, points will be ordered clockwise;

                      if False, points will be ordered counterclockwise.

    :return: A list of (x, y) tuples representing the reordered points.

    """

    # Calculate the centroid of the points

    center_x = np.mean([x for x, y in points])

    center_y = np.mean([y for x, y in points])



    # Define a sorting key function to sort points based on their angle from the centroid

    def sorting_key(point):

        # Calculate the angle of the point relative to the centroid

        angle = -np.arctan2(point[1] - center_y, point[0] - center_x)

        return angle



    # Sort points based on the calculated angle

    sorted_points = sorted(points, key=sorting_key)



    # Return sorted points in the desired order (clockwise or counterclockwise)

    return sorted_points if clockwise else list(reversed(sorted_points))



def simplify_segmentation(points, epsilon=1.0, closed=True):

    """

    Simplifies a curve composed of segmentation points using the Douglas-Peucker algorithm.

    :param points: A list or array of (x, y) coordinates representing the curve.

    :param epsilon: Maximum distance between the original curve and its approximation.

    :param closed: Whether the curve is closed (True) or open (False).

    :return: A list of simplified (x, y) points.

    """

    points_array = np.array(points, dtype=np.float32)

    hull = cv2.convexHull(points_array)

    simplified_hull = cv2.approxPolyDP(hull, epsilon, closed)

    return simplified_hull.reshape(-1, 2)





def appx_best_fit_ngon(points, n=4):

    """

    Approximate the best fit n-gon for a given set of points.



    :param points: A list of (x, y) tuples or a NumPy array of points.

    :param n: The desired number of vertices for the n-gon.

    :return: A list of (x, y) tuples representing the vertices of the approximated n-gon.

    """

    # Calculate the convex hull of the points

    hull = cv2.convexHull(np.array(points, dtype=np.float32)).reshape(-1, 2)

    # Convert points to sympy.Point objects for further geometric processing

    hull = [sympy.Point(*pt) for pt in hull]



    # Iteratively reduce the number of vertices in the convex hull until we reach n vertices

    while len(hull) > n:

        best_candidate = None

        for edge_idx_1 in range(len(hull)):

            # Identify the adjacent vertices for the current edge

            edge_idx_2 = (edge_idx_1 + 1) % len(hull)

            adj_idx_1 = (edge_idx_1 - 1) % len(hull)

            adj_idx_2 = (edge_idx_1 + 2) % len(hull)



            # Create sympy Points for all vertices involved

            edge_pt_1, edge_pt_2 = hull[edge_idx_1], hull[edge_idx_2]

            adj_pt_1, adj_pt_2 = hull[adj_idx_1], hull[adj_idx_2]



            # Form a polygon to calculate the angles at the edge points

            subpoly = sympy.Polygon(adj_pt_1, edge_pt_1, edge_pt_2, adj_pt_2)

            angle1, angle2 = subpoly.angles[edge_pt_1], subpoly.angles[edge_pt_2]



            # Check if the sum of the angles is greater than 180°, otherwise skip this edge

            if sympy.N(angle1 + angle2) <= sympy.pi:

                continue



                # Find the intersection point if we delete the current edge

            intersect = sympy.Line(adj_pt_1, edge_pt_1).intersection(sympy.Line(edge_pt_2, adj_pt_2))[0]

            # Calculate the area of the new triangle that would be formed

            area = sympy.N(sympy.Triangle(edge_pt_1, intersect, edge_pt_2).area)



            # Choose the candidate with the smallest area

            if not best_candidate or area < best_candidate[1]:

                # Create a new hull by replacing the edge with the intersection point

                better_hull = list(hull)

                better_hull[edge_idx_1] = intersect

                del better_hull[edge_idx_2]

                best_candidate = (better_hull, area)



                # Raise an error if no candidate was found (which should not happen with a convex hull)

        if not best_candidate:

            raise ValueError("Could not find the best fit n-gon!")



            # Update the hull with the best candidate found

        hull = best_candidate[0]



        # Convert the final hull points back to integer tuples

    hull = [(int(pt.x), int(pt.y)) for pt in hull]

    return hull





def calculate_distance(x1, y1, x2, y2):

    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return distance





def correct_perspective(image, points):

    # Assuming points are ordered as [top-left, top-right, bottom-right, bottom-left]

    tl, tr, br, bl = points



    # Compute width as average of top and bottom widths

    top_width = calculate_distance(tl[0], tl[1], tr[0], tr[1])

    bottom_width = calculate_distance(bl[0], bl[1], br[0], br[1])

    width = int((top_width + bottom_width) / 2)



    # Compute height as average of left and right heights

    left_height = calculate_distance(tl[0], tl[1], bl[0], bl[1])

    right_height = calculate_distance(tr[0], tr[1], br[0], br[1])

    height = int((left_height + right_height) / 2)





    # Define the target points

    targets = np.array([(0, 0), (width, 0), (width, height), (0, height)], dtype=np.float32)

    corners = np.array(points, dtype=np.float32)



    # Apply perspective transform

    M = cv2.getPerspectiveTransform(corners, targets)

    warped_image = cv2.warpPerspective(image, M, (width, height))



    return warped_image









def find_best_candidate(hull, edge_idx_1):

    """

    Find the best candidate to reduce the number of vertices in the convex hull.



    :param hull: List of sympy.Point objects representing the current convex hull.

    :param edge_idx_1: Index of the current edge point in the hull.

    :return: A tuple (better_hull, area) where better_hull is the new hull and area is the area of the triangle formed,

             or None if no valid candidate is found.

    """

    edge_idx_2 = (edge_idx_1 + 1) % len(hull)

    adj_idx_1 = (edge_idx_1 - 1) % len(hull)

    adj_idx_2 = (edge_idx_1 + 2) % len(hull)



    edge_pt_1, edge_pt_2 = hull[edge_idx_1], hull[edge_idx_2]

    adj_pt_1, adj_pt_2 = hull[adj_idx_1], hull[adj_idx_2]



    subpoly = sympy.Polygon(adj_pt_1, edge_pt_1, edge_pt_2, adj_pt_2)

    angle1, angle2 = subpoly.angles[edge_pt_1], subpoly.angles[edge_pt_2]



    if sympy.N(angle1 + angle2) <= sympy.pi:

        return None



    intersect = sympy.Line(adj_pt_1, edge_pt_1).intersection(sympy.Line(edge_pt_2, adj_pt_2))[0]

    area = sympy.N(sympy.Triangle(edge_pt_1, intersect, edge_pt_2).area)



    better_hull = list(hull)

    better_hull[edge_idx_1] = intersect

    del better_hull[edge_idx_2]

    return (better_hull, area)



def appx_best_fit_ngon2(points, n=4):

    """

    Approximate the best fit n-gon for a given set of points.



    :param points: A list of (x, y) tuples or a NumPy array of points.

    :param n: The desired number of vertices for the n-gon.

    :return: A list of (x, y) tuples representing the vertices of the approximated n-gon.

    """

    # Calculate the convex hull of the points

    hull = cv2.convexHull(np.array(points, dtype=np.float32)).reshape(-1, 2)

    hull = [sympy.Point(*pt) for pt in hull]



    # Iteratively reduce the number of vertices in the convex hull until we reach n vertices

    while len(hull) > n:

        best_candidate = None

        with concurrent.futures.ThreadPoolExecutor() as executor:

            # Submit tasks to find the best candidate for each edge in parallel

            futures = [executor.submit(find_best_candidate, hull, edge_idx_1) for edge_idx_1 in range(len(hull))]

            for future in concurrent.futures.as_completed(futures):

                candidate = future.result()

                if candidate and (not best_candidate or candidate[1] < best_candidate[1]):

                    best_candidate = candidate



        if not best_candidate:

            raise ValueError("Could not find the best fit n-gon!")



        # Update the hull with the best candidate found

        hull = best_candidate[0]



    # Convert the final hull points back to integer tuples

    hull = [(int(pt.x), int(pt.y)) for pt in hull]

    return hull





def main(request_id, image_path, points_1, debug = False):

    """

    Corrects the perspective of the plane in the image, performs OCR, and sends the processed image for further analysis.



    Parameters:

    request_id (str): The ID of the request.

    image_path (str): The path to the image to be processed.

    points_1 (list): A list of points for the initial segmentation.

    debug (bool): Flag to enable debug mode, which sends additional information to RLEF.



    Returns:

    str: Reference number.

    """

    # Read the image from the given path

    image = cv2.imread(image_path)
    points_1 = reorder_points(points=points_1)
    # Simplify the segmentation points

    start_time = time.time()

    simplified_points = simplify_segmentation(points_1, epsilon=20.0, closed=True)

    # print(f"simplify_segmentation {image_path} --- {time.time() - start_time} seconds ---")

    # print(f"simplified points: {simplified_points}")



    # Send simplified points and image to RLEF for debugging

    if debug:

        send_to_rlef.send_segmentation_to_rlef(

            request_id, "backlog", "csv", "66992aad4535d7eef93a6230", "simplified-points",

            request_id, 100, "predicted", "imageAnnotations",

            image_path, "image/png", [simplified_points], [100],

            "simplified_points"

        )



    # Find optimized quadrilateral points

    start_time = time.time()

    points = appx_best_fit_ngon(simplified_points)

    points = reorder_points(points = points, clockwise = False)

    # print(f"find_optimized_quadrilateral_op {image_path} --- {time.time() - start_time} seconds ---")



    # Send corner points to RLEF for debugging

    if debug:

        send_to_rlef.send_segmentation_to_rlef(

            request_id, "backlog", "csv", "66992b1e4535d7eef93a93b9", "corner-points",

            request_id, 100, "predicted", "imageAnnotations",

            image_path, "image/png", [points], [100],

            "corner_points"

        )



    # Correct the perspective of the image

    start_time = time.time()

    plane_corrected_image = correct_perspective(image, points)

    # print(f"plane_correction {image_path} --- {time.time() - start_time} seconds ---")

    # print(type(plane_corrected_image))



    # Save the plane-corrected image

    plane_corrected_image_path = f"debug_plane_other/{os.path.basename(image_path)}"

    # t2 = threading.Thread(target=save_image, args=(plane_corrected_image_path, plane_corrected_image))

    # t2.start()

    save_image(plane_corrected_image_path, plane_corrected_image)



    # Send the plane-corrected image to RLEF

    send_to_rlef.send_image_to_rlef(

        request_id, "backlog", "csv", "669044c9126da81d11c0d599", "plane-corrected-image",

        f"plane-corrected {request_id}", 100, "predicted", "image", plane_corrected_image_path, "image/png", True 

    )
    return plane_corrected_image_path



def read_points_from_file(file_path):
    """
    Reads points from a file and converts them into a list of tuples.
    Each line in the file should contain a point in the format (x,y).
    Parameters:

    file_path (str): The path to the file containing the points.

    Returns:

    list of tuples: A list where each element is a tuple representing a point (x, y).

    """

    points = []

    with open(file_path, 'r') as file:

        lines = file.readlines()

        for line in lines:

            # Remove any leading/trailing whitespace and parentheses

            line = line.strip().strip('()')

            if line:  # Proceed only if the line is not empty

                # Split the line by comma and convert the parts to integers

                x, y = map(int, line.split(','))

                # Append the point as a tuple to the points list

                points.append((x, y))

    return points

