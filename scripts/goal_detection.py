#!/usr/bin/env python

"""Implementation of soccer goal detection
   Goal is represented by 2 orange/red cones (pylons)
"""

# For Python2/3 compatibility
from __future__ import print_function
from __future__ import division

import sys
import os
import math
import rospy
import angles
import tf
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped

import cv2
#print(cv2.__version__)
import numpy as np
import xml.etree.ElementTree as ET


__author__ = "Eric Dortmans"


class GoalDetection:
    """This class detects a soccer goal consisting of 2 training cones (pylons)

    A transform is published from the camera to the goal.
    """

    def __init__(self, camera):
        self.process_image_setup()
        self.bridge = CvBridge()
        self.image_subscriber = rospy.Subscriber("image_raw", Image, self.on_image_message,  
            queue_size=1, buff_size=2**24) # large buff_size for lower latency
        self.transform_publisher = rospy.Publisher("goal", TransformStamped, queue_size=1)
        self.transform_broadcaster = tf.TransformBroadcaster()
               
    def to_cv2(self, image_msg):
        """Convert ROS image message to OpenCV image

        """
        try:
            image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        except CvBridgeError as e:
            print(e)
        return image

    def to_imgmsg(self, image):
        """Convert OpenCV image to ROS image message

        """
        try:
            image_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        except CvBridgeError as e:
            print(e)
        return image_msg

    def on_image_message(self, image_msg):
        """Process received ROS image message
        """
        
        self.image = self.to_cv2(image_msg)
        self.timestamp = image_msg.header.stamp
        self.frame_id = image_msg.header.frame_id
        self.process_image()

    def process_image_setup(self):
        """Setup for image processing. 

        This code will run only once to setup image processing.
        """
        
        self.display = rospy.get_param('~display', True)
        self.goal_frame_id = rospy.get_param('~goal_frame_id', 'goal')
        
        # Optical center coordinates
        (self.center_x, self.center_y) = (None, None)
        
        # Goal coordinates
        (self.goal_x, self.goal_y, self.goal_theta) = (0, 0, 0)
                  

    def process_image(self):
        """Process the image

        This code is run for reach image
        """

        # Size of original image
        width = self.image.shape[1]
        height = self.image.shape[0]
        
        # Make copy of image for display purposes
        display_img = self.image.copy()       

        # Determine optical center
        if self.center_x == None or self.center_y == None:
          Camera.detect_optical_center(self.image) # optional
          self.center_x = int(round(Camera.center[0]))
          self.center_y = int(round(Camera.center[1]))
        
        # Draw crosshair
        north = (self.center_x, height-1)
        south = (north[0], 0)
        east = (width-1, self.center_y)
        west = (0, east[1])
        cv2.line(display_img, south, north, (0,255,0))
        cv2.line(display_img, west, east, (0,255,0))
        cv2.circle(display_img, (self.center_x, self.center_y), 0, (0, 255, 0), 5)  
                
        # Detect soccer training cones (pylons)
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        color_min = np.array([ 0, 150, 150], np.uint8)  # min HSV color
        color_max = np.array([20, 255, 255], np.uint8)  # max HSV color
        blobs = Utils.detect_colored_blobs(hsv, color_min, color_max)  
        #cv2.imshow('blobs', blobs)    
        #pylons = cv2.bitwise_and(self.image, self.image, mask=blobs)
        #cv2.imshow('pylons', pylons)
        contours = Utils.find_contours(blobs)
        if len(contours) > 1: # we have seen then pylons
          cnts = Utils.largest_contours(contours, number=2)
          
          # Calculate and draw position of both goal posts (pylons)
          pylon1 = Utils.center_of_contour(cnts[0])     
          cv2.circle(display_img, pylon1, 0, (0, 255, 0), 5)
          pylon2 = Utils.center_of_contour(cnts[1])
          cv2.circle(display_img, pylon2, 0, (0, 255, 0), 5)
          if pylon1[0] > pylon2[0]:
            pylon1, pylon2 = pylon2, pylon1
          
          # Draw goal-line
          cv2.line(display_img, pylon1, pylon2, (0,255,0), 1)
                   
          # Calculate the center of the goal in pixel coordinates                 
          goal = np.round(Utils.middle_between(pylon1, pylon2)).astype("int")
          goal_x = goal[0]
          goal_y = goal[1]
          
          # Draw line from optical center to goal center
          cv2.circle(display_img, tuple(goal), 0, (0, 0, 255), 5)
          cv2.line(display_img, (self.center_x, self.center_y), tuple(goal), (0,0,255), 1)
            
          # Calculate the goal center real world coordinates
          goal_relative_x = goal_x - self.center_x
          goal_relative_y = goal_y - self.center_y
          goal_rho, goal_phi = Utils.cart2pol(goal_relative_x, goal_relative_y)
          goal_real_phi = goal_phi
          goal_real_rho = Camera.pixels2meters(goal_rho)
          goal_x_cart, goal_y_cart = Utils.pol2cart(goal_real_rho, goal_real_phi)
          self.goal_x, self.goal_y = Camera.image2robot(goal_x_cart, goal_y_cart)

          # Calculate goal orientation (theta)            
          goal_normal_relative = Utils.perpendicular((pylon1[0] - pylon2[0], pylon1[1] - pylon2[1]))
          goal_normal = (goal[0] + goal_normal_relative[0], goal[1] + goal_normal_relative[1])
          cv2.circle(display_img, goal_normal, 0, (0, 255, 0), 5)
          cv2.line(display_img, tuple(goal), tuple(goal_normal), (0,255,0), 1)
          x_axis = (0, -self.center_y)
          self.goal_theta = Utils.angle_between(x_axis, goal_normal_relative)
          #print("goal_theta", self.goal_theta)

        # Publish transform from camera to goal
        transform_msg = TransformStamped()
        transform_msg.header.stamp = rospy.Time.now()
        transform_msg.header.frame_id = self.frame_id
        transform_msg.child_frame_id = self.goal_frame_id
        transform_msg.transform.translation.x = self.goal_x
        transform_msg.transform.translation.y = self.goal_y
        transform_msg.transform.translation.z = 0.0
        quaternion = tf.transformations.quaternion_from_euler(0, 0, self.goal_theta)
        transform_msg.transform.rotation.x = quaternion[0]
        transform_msg.transform.rotation.y = quaternion[1]
        transform_msg.transform.rotation.z = quaternion[2]
        transform_msg.transform.rotation.w = quaternion[3]
        
        self.transform_publisher.publish(transform_msg)
        
        #self.transform_broadcaster.sendTransform(transform_msg)
        self.transform_broadcaster.sendTransform(
          (self.goal_x, self.goal_y, 0.0),
          (quaternion[0], quaternion[1], quaternion[2], quaternion[3]),
          rospy.Time.now(),
          self.goal_frame_id,
          self.frame_id
        )     

        # Show augmented image
        if self.display:
          cv2.imshow("image", display_img)
          cv2.waitKey(1)


class Camera:
    """ Camera parameters
    """
    center = None # Optical center
    
    def __init__(self, params_file):
      root = ET.parse(params_file).getroot()
      # camera/center/x,y
      center_x = float(root.findall("./center/x")[0].text)
      center_y = float(root.findall("./center/y")[0].text)
      Camera.center = np.array([center_x, center_y])
      
      # camera/coefficients/c0,c1,c2,c3
      c0 = float(root.findall("./coefficients/c0")[0].text)
      c1 = float(root.findall("./coefficients/c1")[0].text)
      c2 = float(root.findall("./coefficients/c2")[0].text)
      c3 = float(root.findall("./coefficients/c3")[0].text)
      Camera.coefficients = np.array([c0, c1, c2, c3])

    @classmethod
    def detect_optical_center(cls, image):
        """ detect optical center of omnivision camera
        """
        height, width = image.shape[:2]
        
        # first estimate
        center_x, center_y = width//2, height//2

        # try to give better estimate
        median = cv2.medianBlur(image,5)
        edges = Utils.find_edges(median)
        #cv2.imshow("edges", edges)     
        circles = Utils.find_circles(edges, param2=40, minDist=100, minRadius=180, maxRadius=300)      
        r_max = 0
        if circles is not None:         
          # find biggest circle
          for (x, y, r) in circles:
            if r > r_max:
              r_max = r
              center_x, center_y = x, y    
        
        Camera.center = np.array([center_x, center_y])
        #return center_x, center_y

    @classmethod
    def pixels2meters(cls, rho):
        """ Mapping of image radial distance to real world radial distance.
        """
        polynome = np.array([1, rho, rho**2, rho**3])
              
        return polynome.dot(cls.coefficients)

    @staticmethod
    def image2robot(x, y):
        """ Transform from image to robot coordinates
        
        Image      Robot
        +---x          x
        |              |
        y          y---+       
         
        """
        return -y, -x


class Utils:
    """ Utility methods
    """
    
    @staticmethod
    def cart2pol(x, y):
        """ Carthesian to Polar coordinates
        """
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return (rho, phi)
        
    @staticmethod
    def pol2cart(rho, phi):
        """ Polar to Carthesian coordinates
        """
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    @staticmethod
    def perpendicular(v) :
        """ Vector perpendicular to input vector
        """
        vp = np.empty_like(v)
        vp[0] = -v[1]
        vp[1] =  v[0]       
        return vp

    @staticmethod
    def normalize(v):
        """ Normalize vector to unit magnitude
        """
        v = np.array(v)  
        return v/np.linalg.norm(v)
  
    @staticmethod
    def middle_between(v1, v2) :
        """ Middle between two vector
        """
        v1 = np.array(v1)
        v2 = np.array(v2)
        vm = (v1 + v2) / 2.0
             
        return vm 
               
    @staticmethod
    def angle_between(v1, v2):
        """ Angle between two vectors
        """
        v1 = np.array(v1)
        v2 = np.array(v2)
        
        ## Inner angle, no sign
        #cosang = np.dot(v1, v2)
        #sinang = np.linalg.norm(np.cross(v1, v2))
        #return np.arctan2(sinang, cosang)

        ## Inner angle, no sign
        #v1_u = Utils.normalize(v1)
        #v2_u = Utils.normalize(v2)
        #return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        
        a1 = np.arctan2(v1[1],v1[0])
        a2 = np.arctan2(v2[1],v2[0])
        
        return angles.shortest_angular_distance(a2, a1)
    
    @staticmethod
    def find_contours(image, mode=cv2.RETR_EXTERNAL):
        """find contours in image
        """
        (_, cnts, hierarchy) = cv2.findContours(image.copy(), mode, cv2.CHAIN_APPROX_SIMPLE)

        return cnts

    @staticmethod
    def largest_contours(cnts, number=1):
        """Select largest contour(s)
        """
        largest = sorted(cnts, key=cv2.contourArea, reverse=True)[:number]

        return largest

    @staticmethod
    def center_of_contour(cnt):
        """ Calculate centroid of contour
        """
        moments = cv2.moments(cnt)
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        
        return center_x, center_y
    
    @staticmethod
    def detect_colored_blobs(image, color_min, color_max):
        blobs = cv2.inRange(image, color_min, color_max)
        if (blobs[0, 0] == 255): blobs = cv2.bitwise_not(blobs)
        
        return blobs
        
    @staticmethod
    def find_edges(image):
        """find edges in image
        """
        sigma=0.33
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        return edged
        
    @staticmethod
    def find_circles(image, minDist=1, param1=50, param2=30, minRadius=0, maxRadius=0):
        """Find circles in image
        """

        # detect circles in the image
        circles = cv2.HoughCircles(image.copy(), cv2.HOUGH_GRADIENT, 1,
                                   minDist=minDist,
                                   param1=param1,
                                   param2=param2,
                                   minRadius=minRadius,
                                   maxRadius=maxRadius)
        # convert the (x, y) coordinates and radius of the circles to integers
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            return circles
        
        return circles


def main(args):
    rospy.init_node('goal_detection', anonymous=True)
    
    camera_params = rospy.get_param('~camera_params', None)
    camera = Camera(camera_params)
    
    goal_detection = GoalDetection(camera)
    rospy.spin()
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
