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
import tf
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped

import cv2
#print(cv2.__version__)

import numpy as np

__author__ = "Eric Dortmans"

class GoalDetection:
    """This class detects a soccer goal consisting of 2 orange training cones

    A transform is published from the camera to the goal.
    """

    def __init__(self):
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
        
        # Initial goal estimate
        (self.goal_x, self.goal_y, self.goal_theta) = (0.0, 0.0, 0.0)
                  

    def process_image(self):
        """Process the image using OpenCV DNN

        This code is run for reach image
        """

        # Size of original image
        width = self.image.shape[1]
        height = self.image.shape[0]
        
        # Make copy of image for display purposes
        display_img = self.image.copy()       

        # Detect optical center of camera
        center_x, center_y, r = self.detect_optical_center(self.image)
        cv2.circle(display_img, (center_x, center_y), r, (0, 255, 0), 2)
        cv2.circle(display_img, (center_x, center_y), 0, (0, 0, 255), 5)  
                
        # Detect soccer pylons
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        color_min = np.array([ 0, 150, 150], np.uint8)  # min HSV color
        color_max = np.array([20, 255, 255], np.uint8)  # max HSV color
        blobs = self.detect_colored_blobs(hsv, color_min, color_max)  
        #cv2.imshow('blobs', blobs)    
        #pylons = cv2.bitwise_and(self.image, self.image, mask=blobs)
        #cv2.imshow('pylons', pylons)
        
        # find the contours in the mask
        contours = self.find_contours(blobs)
        if len(contours) > 1: # we have seen some soccer pylons
          cnts = self.largest_contours(contours, number=2)
          pylon1 = self.center_of_contour(cnts[0])     
          cv2.circle(display_img, pylon1, 0, (0, 255, 0), 5)
          pylon2 = self.center_of_contour(cnts[1])
          cv2.circle(display_img, pylon2, 0, (0, 255, 0), 5)
          
          #goal_width = math.hypot(pylon1[0]-pylon2[0], pylon1[1]-pylon2[1])
          #print("goal_width (pixels): ", goal_width)
          
          goal_x = int((pylon1[0]+pylon2[0])/2.0)
          goal_y = int((pylon1[1]+pylon2[1])/2.0)
          cv2.circle(display_img, (goal_x, goal_y), 0, (0, 0, 255), 5)
          cv2.line(display_img, (center_x, center_y), (goal_x, goal_y), (0,0,255))
            
          # Goal coordinates
          goal_rho, goal_phi = self.cart2pol(goal_x-center_x, goal_y-center_y)
          goal_real_phi = goal_phi
          goal_real_rho = self.pixel2world(goal_rho)
          goal_x_cart, goal_y_cart = self.pol2cart(goal_real_rho, goal_real_phi)
          self.goal_x, self.goal_y = self.image2robot(goal_x_cart, goal_y_cart)
        
          ### TODO: calculate goal orientation and store in self.goal_theta


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

    def image2robot(self, x, y):
        """ Transform from image to robot coordinates
        """
        return -y, -x
        
        

    def pixel2world(self, rho):
        """ Calibrated mapping of image distance to real world distance
        """ 
        real_rho = rho / 75 # temporary hack
       
        ## TODO: replace by real lookup/calculation
        
        return real_rho


    def cart2pol(self, x, y):
        """ Carthesian to Polar coordinates
        """
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)
        

    def pol2cart(self, rho, phi):
        """ Polar to Carthesian coordinates
        """
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)


    def find_contours(self, image, mode=cv2.RETR_EXTERNAL):
        """find contours in bw image
        """
        (_, cnts, hierarchy) = cv2.findContours(image.copy(), mode, cv2.CHAIN_APPROX_SIMPLE)

        return cnts


    def largest_contours(self, cnts, number=1):
        """Select largest contour(s)
        """
        largest = sorted(cnts, key=cv2.contourArea, reverse=True)[:number]

        return largest

    def center_of_contour(self, cnt):
        """ Calculate centroid of contour
        """
        moments = cv2.moments(cnt)
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        
        return center_x, center_y
    

    def detect_colored_blobs(self, image, color_min, color_max):
        blobs = cv2.inRange(image, color_min, color_max)
        if (blobs[0, 0] == 255): blobs = cv2.bitwise_not(blobs)
        
        return blobs
        

    def find_edges(self, image):
        """find edges in image
        """
        sigma=0.33
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        return edged
        

    def find_circles(self, image, minDist=1, param1=50, param2=30, minRadius=0, maxRadius=0):
        """Find circles using HoughCircles

        http://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
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
            #circles = np.round(circles[0, :]).astype("int")
            circles = np.uint16(np.around(circles[0,:]))
            return circles
        else:
            return None

    def detect_optical_center(self, image):
        """ detect optical center of omnivision camera
        """
        height, width = image.shape[:2]
        
        # first estimate
        center_x, center_y = width//2, height//2

        # try to give better estimate
        edges = self.find_edges(image)
        #cv2.imshow("edges", edges)     
        circles = self.find_circles(edges, param2=40, minDist=100, minRadius=180, maxRadius=300)      
        image_with_circles = image.copy()
        r_max = 0
        if circles is not None:         
          # find biggest circle
          for (x, y, r) in circles:
            if r > r_max:
              r_max = r
              center_x, center_y = x, y
        
        return center_x, center_y, r_max


def main(args):
    rospy.init_node('goal_detection', anonymous=True)
    ip = GoalDetection()
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
