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
adf
![][image1]

asdf

![][image2]

asdf

![][image3]

asdf

![][image4]

asdf

![][image5]

asdf

![][image6]
