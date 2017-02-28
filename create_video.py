# -*- coding: utf-8 -*-
import glob
import numpy as np
import cv2
from moviepy.editor import VideoFileClip

from utils import calibration_mtx_dist
from utils import get_perspective_map_matrix, get_inverse_perspective_map_matrix
from utils import binarize_img, find_pixel_pos, continue_find_pixel_pos

from collections import deque

 # Choose the number of sliding windows
nwindows = 8
# Set the width of the windows +/- margin
margin = 50
# Set minimum number of pixels found to recenter window
minpix = 50


class line():
    '''
    Storing line attributes
    '''
    def __init__(self):
        self.detected = False
        self.roc = None
        self.fit = None
        self.fit_cr = None
        # Storing up to 3 collection of points from previous frames
        # this help smooth the fitted lines
        self.xs = deque(maxlen=3)
        self.ys = deque(maxlen=3)

    def radius_of_curvature(self, y, fit):
        '''
        Radius of curvature for a second order polynomial
        '''
        return ((1 + (2*fit[0]*y + fit[1])**2)**1.5) / np.absolute(2*fit[0])

    def sanity_check(self, fit, y, x, ym_per_pix, xm_per_pix, ratio):
        '''
        Perform sanity check on radius of curvature
        '''
        if self.fit is None:
            self.fit = fit
            self.fit_cr = np.polyfit(y*ym_per_pix, x*xm_per_pix, 2)
            self.detected = True
            return

        self.roc = self.radius_of_curvature(719, self.fit)

        diff_roc = [np.abs(np.abs(self.roc) - np.abs(self.radius_of_curvature(719, fit))) < ratio*np.abs(self.roc)]

        if all(diff_roc):
            self.xs.append(x)
            self.ys.append(y)
            self.fit = np.polyfit(np.concatenate(self.ys), np.concatenate(self.xs), 2)
            self.fit_cr = np.polyfit(np.concatenate(self.ys)*ym_per_pix, np.concatenate(self.xs)*xm_per_pix, 2)
            self.detected = True
        else:
            self.detected = False

# Initialize left and right lane lines
lLane = line()
rLane = line()

def process_image(img, mtx, dist, M, M_inverse):
    '''
    Pipeline for processing image
    '''
    # convert to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Binarize the image
    binary = binarize_img(img)

    # Distortion correction
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    undist2 = cv2.undistort(binary, mtx, dist, None, mtx)
    img_size = undist2.shape[::-1]
    binary_warped = cv2.warpPerspective(undist2, M, img_size)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/680 # meters per pixel in x dimension

    # Find lanes
    # if both lanes are detected, use fitted lines found in the previous lanes
    if lLane.detected and rLane.detected:
        leftx, lefty, rightx, righty = continue_find_pixel_pos(binary_warped, lLane.fit, rLane.fit)
        if lefty.size > 0:
            left_fit = np.polyfit(lefty, leftx, 2)
            lLane.sanity_check(left_fit, lefty, leftx, ym_per_pix, xm_per_pix, 0.4)
        else:
            lLane.detected = False
        if righty.size > 0:
            right_fit = np.polyfit(righty, rightx, 2)
            rLane.sanity_check(right_fit, righty, rightx, ym_per_pix, xm_per_pix, 0.4)
        else:
            rLane.detected = False
    # if one of the lanes is not detected, relocate the line through sliding windows
    if not(lLane.detected and rLane.detected):
        leftx, lefty, rightx, righty = find_pixel_pos(binary_warped)
        if (not lLane.detected) and lefty.size > 0:
            left_fit = np.polyfit(lefty, leftx, 2)
            lLane.sanity_check(left_fit, lefty, leftx, ym_per_pix, xm_per_pix, 2.)
        if (not rLane.detected) and righty.size > 0:
            right_fit = np.polyfit(righty, rightx, 2)
            rLane.sanity_check(right_fit, righty, rightx, ym_per_pix, xm_per_pix, 2.)

    # Fit a second order polynomial to each
    left_fit_cr = lLane.fit_cr
    right_fit_cr = rLane.fit_cr

    # Calculate the new radii of curvature
    y_eval = (img_size[1]-1)*ym_per_pix
    left_curverad = lLane.radius_of_curvature(y_eval, left_fit_cr)
    right_curverad = rLane.radius_of_curvature(y_eval, right_fit_cr)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = lLane.fit[0]*ploty**2 + lLane.fit[1]*ploty + lLane.fit[2]
    right_fitx = rLane.fit[0]*ploty**2 + rLane.fit[1]*ploty + rLane.fit[2]

    # Calculate off the center distance
    offset = (img_size[0]/2 - (right_fitx[-1] + left_fitx[-1])/2)*xm_per_pix

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Draw information on the frame
    alpha=0.3
    overlay = cv2.warpPerspective(color_warp, M_inverse, img_size)
    undist = cv2.addWeighted(undist, 1, overlay, alpha, 0)
    l_or_r = "left" if offset < 0 else "right"
    cur = 0.5*(left_curverad + right_curverad)
    cv2.putText(undist, "vehicle is {:2.2f}m {} off the center".format(np.abs(offset), l_or_r),
                 (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255,0), 2)
    cv2.putText(undist, "radius of curvature is {:8.2f}m".format(cur),
                 (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)

    return cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)



if __name__ == "__main__":

    images = glob.glob('./camera_cal/calibration*.jpg')
    img = cv2.imread('./test_images/straight_lines1.jpg')

    #find matrices for correcting distortion and perspective transformation
    binary = binarize_img(img)
    mtx, dist = calibration_mtx_dist(images, 9, 6, img.shape[:2][::-1])
    img_size = binary.shape[::-1]
    M = get_perspective_map_matrix(img_size)
    M_inverse = get_inverse_perspective_map_matrix(img_size)

    #process the video
    output = 'project_video_output1.mp4'
    clip2 = VideoFileClip('project_video.mp4')
    process_image2 = lambda x: process_image(x, mtx, dist, M, M_inverse)
    clip = clip2.fl_image(process_image2)
    clip.write_videofile(output, audio=False)

