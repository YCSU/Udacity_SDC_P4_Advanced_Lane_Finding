# Udacity_SDC_P4_Advanced_Lane_Finding
Using Sobel filter and color space to find lane lines

[//]: # (Image References)
[image1]: ./output_images/distortion_correction.jpg
[image2]: ./output_images/undist.jpg
[image3]: ./output_images/binarized.jpg
[image4]: ./output_images/warp.jpg
[image5]: ./output_images/fit_lines.jpg
[image6]: ./output_images/outupt.jpg

This is the fourth project for Udacity Self-Driving Car Engineer Nanodegree. For this project, we need to detect the lane lines on the road, and calculate both the radius of curvature and the distance off the center between the lane lines.

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


### Running the code
The projects includes the following files:
* create_video.py - pipeline for detecting lane lines. See the comments in the the file for explanation  
* utils.py - functions including camera calibration, perspective tranform, sobel filter, and finding lane lines
* project_video_output.mp4 - the output video with detected lane lines
* challenge_video_output.mp4 - a not so successful attempt for challenge video
* README.md - the file reading now
Make sure Opencv, moviepy and glob is installed before running the script.

To launch the script, 
```
python create_video.py
```


## Compute the camera calibration matrix and distortion coefficients
For camera calibration (line 14-39 in utils.py), I assume that the (x, y, z) coordinates of the chessboard corners for the objpoints are fixed on the x-y plane at z=0. Everytime the coreners are found in a calibration image, the objpoints and the (x, y) coordinates of the corners in the image are appended to objpoints and imgpoints. Finally, we feed objpoints and imgpoints into cv2.calibrateCamera to obtain the camera calibration matrix and distortion coefficients. Using cv2.undistort(), we obtain the following result (line 280-294 in utils.py):

![][image1]

## Pipeline (single imagle)
The whole pipeline is in the process_image() function (line 68-152 in create_video.py).

### Correct distortion 
After obtaining the camera calibration matrix and distortion coefficients from the previous step, we can use cv2.undistort() to correct the distortion (line 79-80 in create_video.py). Here is an example from the test image:

![][image2]

### Binarzing the image
We apply Sobel filters and thresholdings in r,g,b,saturation colorspace (binarize_img(), line 122-154 in utils). Sobel filters in x, y, magnitude and direction is applied to each color channel. The simple threshholdings was also used to search the candidates.

![][image3]

### Perspective transform
For the perspective trasfrom (line 41-70 in utils.py), the following source and destination points are used for cv2.getPerspectiveTransform() to obtain the transformation matrix. After that, we can use cv2.warpPerspective() to get a "birds-eye view":
```
src = np.float32([(256,678),
                  (1051, 678),
                  (688,450),
                  (593, 450)])

dst = np.float32([[300, 720],
                  [980, 720],
                  [300, 0],
                  [980, 0]])
```
This choice of points is verified by plotting the results of the transformation:

![][image4]

### Fitting the lane lines


![][image5]

### Example output of the pipeline

![][image6]
