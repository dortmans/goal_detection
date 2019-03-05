##############################################################################
#
# script:  calibrate_camera
# project: Fontys Hogescholen ICT RIF robot football
# author:  Peter Lambooij
# version: 1.0
# date:    Dec 4 2018
# purpose: to find calibration parameters of camera and to take calibration snapshots
#
##############################################################################
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import threading
import math

RED = (  0,   0, 255)
DARK_RED = (  0,   0,  50)
max_nr_points = 250
min_quality = 0
min_distance = 0

def get_parameter(default, message):
    question = message + ' (' + str(default) + '): '
    answer = raw_input(question)
    if answer != '':
        return type(default)(answer)  # conversion to correct type
    else:
        return default  # on empty answer return default


def save_frame(frame, file_name):
    cv2.imwrite(file_name, frame)
    print 'saved frame to file "' + file_name + "'"


def change_max_points(nr):
    global max_nr_points
    if distance > 0:
        max_nr_points = nr


def change_min_quality(quality):
    global min_quality
    if quality > 0:
        min_quality = quality


def change_min_distance(distance):
    global min_distance
    if distance > 0:
        min_distance = distance


def change_tile_dist(dist):
    global tile_dist
    if dist > 0:
        tile_dist = dist


def change_tile_size(size):
    global tile_size
    if size > 0:
        tile_size = size


def convert_pixels_to_meter(x, y, xc, yc, c0, c1, c2, c3):
    p = math.sqrt((x - xc)**2 + (y - yc)**2) # use pythagoras to calculate distance in pixels
    d = c0 + c1*p + c2*p**2 + c3*p**3
    return d


def report_quality_of_fit(centerX, centerY, coeffs, x, y):
    print 'd is distance (in m) from robot.'
    print 'p is distance in pixels from center of image'
    print 'center of camera image is (xc, yc)' 
    print 
    print 'fitted points to polynomial:'
    print '    p = sqrt( (x - xc)^2 + (y - yc)^2 )' 
    print '    d = c0 + c1*p + c2*p^2 + c3*p^3'
    print
    print 'with:'
    print '    xc =', centerX
    print '    yc =', centerY
    print '    c0 =', coeffs[-1] # coeffs are from highest to lowest
    print '    c1 =', coeffs[-2] # coeffs are from highest to lowest
    print '    c2 =', coeffs[-3] # coeffs are from highest to lowest
    print '    c3 =', coeffs[-4] # coeffs are from highest to lowest
    print
    print 'quality of the fit:'
    print
    print 'pixels'.rjust(12), 'dist in m'.rjust(12), 'fitted in m'.rjust(12), 'diff (cm)'.rjust(12) 
    mean_square = 0
    for (vx, vy) in zip(x, y):
        py = coeffs[-1] + coeffs[-2]*vx + coeffs[-3]*vx**2 + coeffs[-4]*vx**3
        diff = py - vy
        mean_square += diff**2
        print '{:12.1f}'.format(vx), '{:12.3f}'.format(vy), '{:12.3f}'.format(py), '{:12.1f}'.format(diff*100)
    rms = math.sqrt(mean_square / len(x))  * 100.0
    print 'rms of errors =', '{:8.1f}'.format(rms), 'cm'


def save_calibration(centerX, centerY, coeffs):
    file = open('camera_parameters.xml','w')
    file.write('<?xml version="1.0" encoding="UTF-8"?>\n') 
    file.write('<camera>\n')
    file.write('  <center>\n')
    file.write('    <x>' + str(centerX) + '</x>\n')
    file.write('    <y>' + str(centerY) + '</y>\n')
    file.write('  </center>\n')
    file.write('  <coefficients>\n')
    file.write('    <c0>' + str(coeffs[-1]) + '</c0>\n')
    file.write('    <c1>' + str(coeffs[-2]) + '</c1>\n')
    file.write('    <c2>' + str(coeffs[-3]) + '</c2>\n')
    file.write('    <c3>' + str(coeffs[-4]) + '</c3>\n')
    file.write('  </coefficients>\n')
    file.write('</camera>')
    file.close() 
    print 'center coordinates and coefficients written to calibration.xml'


def plot_calibration(x, y, coeffs):
    xp   = np.linspace(x[0], x[-1], 100)
    yp = []
    for vx in xp:
        vy = coeffs[-1] + coeffs[-2]*vx + coeffs[-3]*vx**2 + coeffs[-4]*vx**3
        yp.append(vy)

    plt.plot(x, y, 'o', xp, yp, '-')
    plt.ylim(0, max(y)*1.2)
    plt.xlim(0, max(x)*1.2)
    plt.title('camera calibration curve')
    plt.xlabel('distance in pixels')
    plt.ylabel('distance in meter')
    plt.show()


def fit_calibration(points, centerX, centerY, tile_dist, tile_size):
    if points != []:
        d0 = tile_dist / 100.0 # distance to first tile
        d1 = tile_size / 100.0
        x = []
        y = []
        for n, p in enumerate(points):
            x.append(p[1] - centerY)
            y.append(d0 + n * d1)
        coeffs = np.polyfit(x, y, 3)
        poly = np.poly1d(coeffs)
        report_quality_of_fit(centerX, centerY, coeffs, x, y)
        save_calibration(centerX, centerY, coeffs)
        plotting_thread = threading.Thread(target=plot_calibration,args=(x, y, coeffs))
        plotting_thread.daemon = True
        plotting_thread.start() # threading to keep both image window and plot window alive


def find_center(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=50)
    return circles[0][0]


def show_center(frame, centerX, centerY, centerR):
    # overlay center and cross-hair
    h = frame.shape[0]
    w = frame.shape[1]
    cv2.line(frame, pt1=(centerX, 0), pt2=(centerX, h-1),
             color=(255, 0, 0), thickness=1, lineType=8, shift=0)
    cv2.line(frame, pt1=(0, centerY), pt2=(w-1, centerY),
             color=(255, 0, 0), thickness=1, lineType=8, shift=0)
    cv2.circle(frame, (centerX, centerY), centerR, (255, 0, 0), 1)


def detect_points(frame, centerX, centerY):
    global max_nr_points, min_distance, min_quality
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(
        gray, max_nr_points, min_quality / 100.0, min_distance)
    corners = np.int0(corners)
    points = []
    for i in corners:
        x, y = i.ravel()
        if abs(x - centerX) < min_distance and y > centerY:
            points.append((x, y))
            cv2.circle(frame, (x-1, y-1), 3, RED, -1)
        else:
            cv2.circle(frame, (x-1, y-1), 3, DARK_RED, -1)
    points.sort(key=lambda x: x[1])
    return points


def show_menu(camera):
    if camera != None:
        print "Move/Rotate the chessboard pattern into the desired position."
        print
    print "Click on the image and press:"
    print "'c' to toggle calibration data on/off"
    print "'f' to fit and plot a calibration curve"
    print "'s' to save a frame to file."
    print "'q' to quit"
    print


def select_input():
    choice, camera, file_name, frame = None, None, None, None

    answer = get_parameter('image.bmp', "input: 'camera', 'quit' or file name")

    if answer == 'quit':
        choice = 'quit'
    elif answer == 'camera':
        choice = 'camera'
        camera_id = get_parameter(0, 'camera id')
        camera = cv2.VideoCapture(camera_id)
        if camera.isOpened():
            success, frame = camera.read()
            print "opened camera with id", camera_id, "\n"
        else:
            print "could not open camera with id", camera_id, "\n"
    else:
        choice = 'file'
        file_name = answer
        frame = cv2.imread(file_name)

    if choice != 'quit':
        cv2.imshow('calibration frame', frame)

    return choice, camera, file_name, frame

def main():
    print
    print "****************************************"
    print "***                                  ***"
    print "*** camera snapshots and calibration ***"
    print "***                                  ***"
    print "****************************************"
    print

    choice, camera, file_name, frame = select_input()

    if choice == 'quit' or (choice == 'camera' and camera == None):
        return # filed to select valid input

    show_menu(camera)
    calibrate = False
    key = 255
    while key != ord('q') and key != 27:  # ord(esc) == 27
        if camera != None:
            success, original_image = camera.read()
        else:
            original_image = cv2.imread(file_name)

        frame = original_image.copy()

        if calibrate:
            centerX, centerY, centerR = find_center(frame)
            show_center(frame, centerX, centerY, centerR)
            points = detect_points(frame, centerX, centerY)

        cv2.imshow('calibration frame', frame)

        key = cv2.waitKey(50) & 0xFF
        if key == ord('q') or key == 27:
            cv2.destroyAllWindows()
        elif key == ord('c'):
            calibrate = not calibrate
            if calibrate:
                global max_nr_points, min_distance, min_quality
                min_distance = 7
                min_quality = 15
                tile_dist = 25
                tile_size = 25
                cv2.createTrackbar('minimum distance (in pixels)', 'calibration frame', 
                    min_distance, 20, change_min_distance)
                cv2.createTrackbar('quality corner',   'calibration frame',
                    min_quality, 100, change_min_quality)
                cv2.createTrackbar('closest point (in cm)', 'calibration frame', 
                    tile_dist,  50, change_tile_dist)
                cv2.createTrackbar('distance between points (in cm)', 'calibration frame', 
                    tile_size,  50, change_tile_size)
            else:
                cv2.destroyAllWindows()
                cv2.imshow('calibration frame', frame)
        elif key == ord('s'): # save the original image
            file_name = get_parameter('image.bmp', 'enter file name')    
            save_frame(original_image, file_name)
            key = cv2.waitKey(50) & 0xFF
        elif key == 82 or key == ord('u'):  # up
            h = frame.shape[0]
            centerY = (centerY - 1) % h
        elif key == 84 or key == ord('d'):  # down
            h = frame.shape[0]
            centerY = (centerY + 1) % h
        elif key == 81 or key == ord('l'):  # left
            w = frame.shape[1]
            centerX = (centerX - 1) % w
        elif key == 83 or key == ord('r'):  # right
            w = frame.shape[1]
            centerX = (centerX + 1) % w
        elif key == ord('f'):
            if calibrate:
                save_frame(original_image, 'camera_original.bmp')
                save_frame(frame, 'camera_calibrated.bmp')
                fit_calibration(points, centerX, centerY, tile_dist, tile_size)
        else:
            if key != 255: # nothing pressed: ignore
                #print 'unknown key: ' + str(key)
                continue  

    if camera != None:
        camera.release()

if __name__ == "__main__":
    main()
