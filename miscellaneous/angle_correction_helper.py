import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
from math import atan2, cos, sin, sqrt, pi

class ImageOrientationCorrection:
    def __init__(self,angle = 0):
      self.angle = angle
      pass

    def drawAxis(self,img, p_, q_, colour, scale):
      ''' This functions 4 parameters
      1: img: image file
      2: p_ : 1st set of points where you want to draw axis
      3: q_ : 2nd set of points where you want to draw axis
      4: colour: a tuple of color in which you want to draw axis
      5: scale : thickness of the line you want to draw
      '''
      self.img = img
      self.p_ = p_
      self.q_ = q_
      self.colour = colour
      self.scale = scale

      p = list(self.p_)
      q = list(self.q_)
      angle = atan2(p[1] - q[1], p[0] - q[0])
      hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

      q[0] = p[0] - self.scale * hypotenuse * cos(angle)
      q[1] = p[1] - self.scale * hypotenuse * sin(angle)
      cv.line(self.img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), self.colour, 1, cv.LINE_AA)

      p[0] = q[0] + 9 * cos(angle + pi / 4)
      p[1] = q[1] + 9 * sin(angle + pi / 4)
      cv.line(self.img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), self.colour, 1, cv.LINE_AA)

      p[0] = q[0] + 9 * cos(angle - pi / 4)
      p[1] = q[1] + 9 * sin(angle - pi / 4)
      cv.line(self.img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), self.colour, 1, cv.LINE_AA)

    def getOrientation(self,pts, img):
      '''
      This function takes 2 parameters
      1: pts: points in which we need to find orientation
      2: img: image file
      '''
      self.pts = pts
      self.img = img
      sz = len(self.pts)
      data_pts = np.empty((sz, 2), dtype=np.float64)
      for i in range(data_pts.shape[0]):
          data_pts[i,0] = pts[i,0,0]
          data_pts[i,1] = pts[i,0,1]
      mean = np.empty((0))
      mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
      m = cv.moments(pts)
      if m['m00'] != 0:
          cx = int(m['m10']/m['m00'])
          cy = int(m['m01']/m['m00'])
      cntr = (cx,cy)
      cv.circle(self.img, cntr, 3, (255, 0, 255), 2)
      p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
      p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
      self.drawAxis(self.img, cntr, p1, (0, 255, 0), 1)
      self.drawAxis(self.img, cntr, p2, (255, 255, 0), 5)

      self.angle = atan2(eigenvectors[1,0], eigenvectors[1,1])
      self.angle = - np.rad2deg(self.angle)
      cv.imwrite("output.png",self.img)

      return self.angle


    def measure_angle(self,src):
      """
        src : np.array type image 
      """
      bw = src 
      contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
      l = []

      for i, c in enumerate(contours):
          area = cv.contourArea(c)
          l.append([i,area,c])
      df = pd.DataFrame(l, columns=['index', 'Area',"contour"])
      df = df.sort_values("Area",ascending=False)
      df.reset_index(inplace =True)
      n = df["index"][0]
      
      cv.drawContours(src, contours, n, (0, 0, 255), 2)
      
      c = df["contour"][0]
      
      self.getOrientation(c, src)
      
      self.angle = round(float(self.getOrientation(c,src)),2)
      print("angle=======",self.angle)
      rotated_rect = cv.minAreaRect(c)
      
      self.angle = rotated_rect[-1]
      print("angle=======",self.angle)

      if self.angle < -45:
        self.angle += 90

      return self.angle

    def all_angles(self,mask,image, save_images = False ):
       """
            There can be two possible orientations : Upside down and Correct orientation. 
       """
       angle = self.measure_angle(mask)
       return [-angle, 180-angle, -angle+90, -angle+270]
       
    def correct_rotation(self,input_image_path,output_image_path,fill):
      ''' This function takes 2
      arguments
          1: input_image_path : The path of the input image
          2: output_image_path: The path of the rotated image

      '''
      self.input_image_path = input_image_path
      self.output_image_path = output_image_path
      self.fill = fill
      image = Image.open(self.input_image_path)
      angle = self.measure_angle(self.input_image_path)

      rotated_image = image.rotate(angle, expand = False, fillcolor = self.fill)
      rotated_image.save(self.output_image_path)
