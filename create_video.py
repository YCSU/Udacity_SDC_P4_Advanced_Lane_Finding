# -*- coding: utf-8 -*-
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

from utils import calibration_mtx_dist, binarize_img, get_perspective_map_matrix, get_inverse_perspective_map_matrix
from collections import deque

 # Choose the number of sliding windows
nwindows = 8
# Set the width of the windows +/- margin
margin = 50
# Set minimum number of pixels found to recenter window
minpix = 50


class line():
    def __init__(self):
        self.detected = False
        self.roc = None
        self.fit = None
        self.fit_cr = None
        self.xs = deque(maxlen=3)
        self.ys = deque(maxlen=3)

    def radius_of_curvature(self, y, fit):
        return ((1 + (2*fit[0]*y + fit[1])**2)**1.5) / np.absolute(2*fit[0])

    def sanity_check(self, fit, y, x, ym_per_pix, xm_per_pix, ratio):
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


lLane = line()
rLane = line()



def continue_find_pixel_pos(binary_warped, left_fit, right_fit):
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    idx = nonzeroy > binary_warped.shape[0]/3
    nonzerox = np.array(nonzero[1])[idx]
    nonzeroy = nonzeroy[idx]
    #margin = 50
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def find_pixel_pos(binary_warped):
    # Set height of windows
    window_height = np.int(2*binary_warped.shape[0]/3/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[2*binary_warped.shape[0]/3:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    win_y_low = binary_warped.shape[0] - window_height
    win_y_high = binary_warped.shape[0]
    win_xleft_low = leftx_base - margin
    win_xleft_high = leftx_base + margin
    win_xright_low = rightx_base - margin
    win_xright_high = rightx_base + margin
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]



    # If you found > minpix pixels, recenter next window on their mean position
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.median(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:
        rightx_current = np.int(np.median(nonzerox[good_right_inds]))



    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    old_left_offset = 0
    old_right_offset = 0
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        good_left_y = nonzeroy[good_left_inds]
        good_right_y = nonzeroy[good_right_inds]
        good_left_x = nonzerox[good_left_inds]
        good_right_x = nonzerox[good_right_inds]
        left_ymedian = np.median(good_left_y)
        right_ymedian = np.median(good_right_y)
        left_offset = np.median(good_left_x[good_left_y > left_ymedian]) - np.median(good_left_x[good_left_y < left_ymedian])
        right_offset = np.median(good_right_x[good_right_y > right_ymedian]) - np.median(good_right_x[good_right_y < right_ymedian])

        # If you found > minpix pixels, recenter next window on their mean position
        if (len(good_left_inds) > minpix) and (not np.isnan(left_offset)) and ((np.sign(left_offset) == np.sign(old_left_offset)) or old_left_offset==0):
            leftx_current = np.int(np.median(good_left_x) - left_offset*1.)
            old_left_offset = np.int(left_offset*1.)
        else:
            leftx_current -= old_left_offset
        if (len(good_right_inds) > minpix) and (not np.isnan(right_offset)) and ((np.sign(right_offset) == np.sign(old_right_offset)) or old_right_offset==0):
            rightx_current = np.int(np.median(good_right_x) - right_offset*1.)
            old_right_offset = np.int(right_offset*1.)
        else:
            rightx_current -= old_right_offset


    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def process_image(img2, mtx, dist, M, M_inverse):


    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    #cv2.imwrite("./test_images/challenge.jpg", img2)

    #img2 = cv2.imread('./test_images/straight_lines2.jpg')
    binary = binarize_img(img2)
    #plt.imshow(binary, cmap='gray')


    undist = cv2.undistort(img2, mtx, dist, None, mtx)
    undist2 = cv2.undistort(binary, mtx, dist, None, mtx)
    img_size = undist2.shape[::-1]
    binary_warped = cv2.warpPerspective(undist2, M, img_size)
    #plt.imshow(binary_warped, cmap="gray")

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/680 # meters per pixel in x dimension


    #try:
    if lLane.detected and rLane.detected:# and (lLane.fit is not None) and (rLane.fit is not None):
        #, out_img, left_lane_inds, right_lane_inds, nonzerox, nonzeroy
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
    if not(lLane.detected and rLane.detected):
        #, out_img, left_lane_inds, right_lane_inds, nonzerox, nonzeroy
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


    alpha=0.3
    #color_warp = cv2.cvtColor(color_warp, cv2.COLOR_BGR2RGB)
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
    img2 = plt.imread('./test_images/test4.jpg')
    #img2 = cv2.imread('./test_images/straight_lines2.jpg')
    #binary = binarize_img(img2)
    #plt.imshow(binary, cmap='gray')


    binary = binarize_img(img2)
    mtx, dist = calibration_mtx_dist(images, 9, 6, img.shape[:2][::-1])
    #undist = cv2.undistort(img, mtx, dist, None, mtx)
    img_size = binary.shape[::-1]
    M = get_perspective_map_matrix(img_size)
    M_inverse = get_inverse_perspective_map_matrix(img_size)


    plt.imshow(binary, cmap='gray')
    #img = process_image(img2, mtx, dist, M, M_inverse)
    #plt.imshow(binary, cmap='gray')
    #undist = cv2.undistort(img, mtx, dist, None, mtx)
    #undist2 = cv2.undistort(binary, mtx, dist, None, mtx)
    #img_size = undist2.shape[::-1]
    #binary_warped = cv2.warpPerspective(undist2, M, img_size)
    #cv2.imwrite("./output_images/warped.jpg", binary_warped)
    #plt.imshow(binary_warped, cmap="gray")
    #plt.savefig("./output_images/binary.jpg")
    #plt.imshow(process_image(img2, mtx, dist, M, M_inverse))
    #plt.savefig("./output_images/outupt.jpg", dpi=150)


    #output = 'challenge_video_output.mp4'
    #clip2 = VideoFileClip('challenge_video.mp4')
    #process_image2 = lambda x: process_image(x, mtx, dist, M, M_inverse)
    #clip = clip2.fl_image(process_image2)
    #clip.write_videofile(output, audio=False)

