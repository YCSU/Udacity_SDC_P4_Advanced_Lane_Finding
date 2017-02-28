# -*- coding: utf-8 -*-
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

 # Choose the number of sliding windows
nwindows = 8
# Set the width of the windows +/- margin
margin = 50
# Set minimum number of pixels found to recenter window
minpix = 50


def calibration_mtx_dist(image_paths, nx, ny, img_size):
    objp = np.zeros((ny*nx, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(image_paths):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return mtx, dist

def get_perspective_map_matrix(img_size):
    src = np.float32([(256,678),
                    (1051, 678),
                    (688,450),
                    (593, 450)
                    ])
    x_offset = 300
    y_offset = 0
    dst = np.float32([[x_offset, img_size[1]-y_offset],
                      [img_size[0]-x_offset,img_size[1]-y_offset],
                      [img_size[0]-x_offset, y_offset],
                      [x_offset, y_offset]])
    return cv2.getPerspectiveTransform(src, dst)

def get_inverse_perspective_map_matrix(img_size):
    src = np.float32([(256,678),
                    (1051, 678),
                    (688,450),
                    (593, 450)
                    ])
    x_offset = 300
    y_offset = 0
    dst = np.float32([[x_offset, img_size[1]-y_offset],
                      [img_size[0]-x_offset,img_size[1]-y_offset],
                      [img_size[0]-x_offset, y_offset],
                      [x_offset, y_offset]])
    return cv2.getPerspectiveTransform(dst, src)


def abs_sobel_thresh(gray, orient='x', thresh_min=0, thresh_max=255):
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return binary_output



def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = (sobelx**2 + sobely**2)**0.5
    scale_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scale_sobel)
    binary_output[(scale_sobel > mag_thresh[0]) & (scale_sobel < mag_thresh[1])] = 1

    return binary_output



def dir_thresh(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    angle = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(angle)
    binary_output[(angle > thresh[0]) & (angle < thresh[1])]=1

    return binary_output

def bgr_thresh(img, thresh=(0,255), channel=2):
    R = img[:,:,channel]
    binary_output = np.zeros_like(R)
    binary_output[(R > thresh[0]) & (R <= thresh[1])] = 1
    return binary_output

def hls_thresh(img, thresh=(0,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    S = hls[:,:,2]
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary_output

def sobel_filterts(img):
    sobelx = abs_sobel_thresh(img, 'x', 70, 255)
    sobely = abs_sobel_thresh(img, 'y', 70, 255)
    sobel_mag = mag_thresh(img, mag_thresh=(70,255))
    sobel_dir = dir_thresh(img, sobel_kernel=15 ,thresh=(0.75, 1.2))
    return sobelx, sobely, sobel_mag, sobel_dir

def binarize_img(img):
    s = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,2])
    r = cv2.equalizeHist(img[:,:,2])
    g = cv2.equalizeHist(img[:,:,1])
    b = cv2.equalizeHist(img[:,:,0])

    sSobelx, sSobely, sSobel_mag, sSobel_dir = sobel_filterts(s)
    gSobelx, gSobely, gSobel_mag, gSobel_dir = sobel_filterts(g)
    rSobelx, rSobely, rSobel_mag, rSobel_dir = sobel_filterts(r)
    bSobelx, bSobely, bSobel_mag, bSobel_dir = sobel_filterts(b)

    r_binary = bgr_thresh(img, (60, 255), 2)
    g_binary = bgr_thresh(img, (60, 255), 1)
    b_binary = bgr_thresh(img, (60, 255), 0)
    s_binary = hls_thresh(img, (90, 255))

    binary = np.zeros_like(s)
    binary[   ((sSobel_dir==1) & (sSobel_mag==1))
            | ((sSobelx==1) & (sSobely==1))
            | ((rSobel_dir==1) & (rSobel_mag==1))
            | ((rSobelx==1) & (rSobely==1))
            | ((gSobel_dir==1) & (gSobel_mag==1))
            | ((gSobelx==1) & (gSobely==1))
            | ((bSobel_dir==1) & (bSobel_mag==1))
            | ((bSobelx==1) & (bSobely==1))
            | ((r_binary==1) & (s_binary==1))
            | ((g_binary==1) & (s_binary==1))
            | ((b_binary==1) & (s_binary==1))]=1
    return binary


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
            leftx_current = np.int(np.median(good_left_x) - left_offset)
            old_left_offset = np.int(left_offset)
        else:
            leftx_current -= old_left_offset
        if (len(good_right_inds) > minpix) and (not np.isnan(right_offset)) and ((np.sign(right_offset) == np.sign(old_right_offset)) or old_right_offset==0):
            rightx_current = np.int(np.median(good_right_x) - right_offset)
            old_right_offset = np.int(right_offset)
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

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    plt.figure(figsize=(8, 6))
    plt.subplot(111)
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.savefig("./output_images/fit_lines.jpg", dpi=80)


if __name__ == "__main__":

    images = glob.glob('./camera_cal/calibration*.jpg')
    img = cv2.imread('./camera_cal/calibration1.jpg')
    mtx, dist = calibration_mtx_dist(images, 9, 6, img.shape[:2][::-1])
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    undist2 = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.subplot(121)
    plt.title("original")
    plt.imshow(img)
    plt.subplot(122)
    plt.title("undistorted")
    plt.imshow(undist2)
    plt.savefig("./output_images/distortion_correction.jpg", dpi=100)


    img = cv2.imread('./test_images/test4.jpg')
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    undist = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.subplot(121)
    plt.title("original")
    plt.imshow(img)
    plt.subplot(122)
    plt.title("undistorted")
    plt.imshow(undist, cmap="gray")
    plt.savefig("./output_images/undist.jpg", dpi=100)


    binarized = binarize_img(undist)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.subplot(121)
    plt.title("original")
    plt.imshow(img)
    plt.subplot(122)
    plt.title("binarized")
    plt.imshow(binarized, cmap="gray")
    plt.savefig("./output_images/binarized.jpg", dpi=100)


    img = cv2.imread('./test_images/straight_lines1.jpg')
    src = np.int_([[(256,678),
                    (1051, 678),
                    (688,450),
                    (593, 450)
                    ]])
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    undist = cv2.polylines(undist, src, True, (0,0,255), 2)
    im_size = img.shape[1], img.shape[0]
    M = get_perspective_map_matrix(im_size)
    warp = cv2.warpPerspective(undist, M, im_size)
    undist = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
    warp = cv2.cvtColor(warp, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.subplot(121)
    plt.title("original")
    plt.imshow(undist)
    plt.subplot(122)
    plt.title("warped")
    plt.imshow(warp)
    plt.savefig("./output_images/warp.jpg", dpi=100)

    img = cv2.imread('./test_images/test5.jpg')
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    binarized = binarize_img(undist)
    binarized_warp = cv2.warpPerspective(binarized, M, im_size)
    find_pixel_pos(binarized_warp)


