# **Project 4 - Advanced Line Finding**


## Project Definition

##### This project is from the fourth part of Udacity's Self Driving Car Nanodegree Program and the goal is to create a pipeline that detects the lane lines robustly in given images and videos. Additional to first project this pipeline is able to detect not only straight lines but also polynomial lines. The curvature and the deviation of the car from lane middle is also printed on each frame.
---
### Project Folder
Short overview about which file contains what:
* Advanced_Lane_Finding.ipynb: Jupyter Notebook that includes my code/pipeline
* Advanced_Lane_Finding.html: HTML version of my Jupyter Notebook
* camera_cal - Camera calibration images
* project_video_short, challenge_video, harder_challenge_video: Video to test the pipeline
* output_images: Images that will be used in this writeup
* output_videos: Output images after processing them with my Pipeline
* test_images: includes some harder frames that I tested my pipeline with
---
###  Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---
### Aftermath - What I have used?
Here is a list of API functions that I have been using along this project.  

* **glob.glob('../testImage*.jpg')** -> returns the name of the images that fits to the String name convention between parenthesis
* **np.mgrid[x1:x2,y1:y2]** -> returns a dense multi-dimensional “meshgrid”.
* **cv2.findChessboardCorners** -> On a given gray Image finds the cornes where black & white pixels changes significantly
* **cv2.drawChessboardCorners** -> draw lines on found corners by findChessboardCorners method
* **cv2.calibrateCamera** -> Calibrates camera given the object (3D) and Image(2D) points
* **cv2.undistort** -> Using camera matrix found by camera calibration, corrects the distorted Image
* **cv2.Sobel** -> Applie sobel operator to Image in order to find edges
* **cv2.getPerspectiveTransform** -> Given source and destination points, retrieves the perspective transform matrix
* **cv2.warpPerspective** -> By using matrix from perspective transform, it warps the image to given destination points
* **cv2.threshold** -> Uses threshold on B&W images, pixel values that are not between given threshold will be blacked out
* **cv2.rectangle** ->Draws rectangle around given pixels
* **cv2.putText** -> Writes text to a given Image
* **np.polyfit** -> Given 2D points and degrees of Polynomial returns the coefficients of polynomial function.
* **np.linspace** -> returns consecutive numbers between given range
* **ffmpeg_extract_subclip** -> cuts given video and creates a new one given in range of second

---

### Camera Calibration

Camera Calibration is necessary to correct distorted images that might be caused by cameras lens. We need to un-distort each frames so we can trust getting straight lines when we warp the perspective, so that correct polynomial would be fit to them.

In order to do that we need camera parameters which can be achieved by calibration procedure with the help of a chessboard texture. We will assume that the corners of chessboard are straightly connected and opencv algorithms will try to find rectification parameters by this assumption.

Camera parameters are achieved in `Advanced_Lane_Finding.ipynb` between blocks 2 - 8. In order to get a robust calibration procedure we would need more than one image. In block 2 I save all the calibration images in to a list. Then I get the ones that have 9x6 amount of corners by applying `findChessboardCorners` function. If an image found then I save corner point coordinates to imagePoint List (2D) and objects points (3D) to objectPoint List. Here Image points are real found coordinates of the corners in an image, which is therefor 2D. Object Images are simply 3D image coordinated that are created with a depth value of 0 (z) and x , y coordinates of each permutation of (0-8,0-5) which will represent the point coordinates in real world after calibration process ex: (0,0,0),(1,0,0)...(8,5,0).

Here are found corners, Image Points, that is detected in one of the calibration images.

![alt text][foundCor]

Then these two lists are given to `calibrateCamera` function with image dimension. So that the Distortion Coefficients and camera Matrix could be retrieved. As a result Camera matrix is found as:
~~~~
[[1.15662906e+03 0.00000000e+00 6.69041438e+02]
 [0.00000000e+00 1.15169194e+03 3.88137239e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
~~~~
and distortion coefficients:
~~~~
[[-0.23157149 -0.1200054  -0.00118338  0.00023305  0.15641575]]
~~~~

As a result applying these parameters to `cv2.undistort` I got the following un-distorted images:

![alt text][comboCalib2]

### Pipeline for each frame

#### 1. Un-distort each frame.

Before starting processing images we have to rectify each image with the parameters found in calibration process. Here, of course the camera which is calibrated and used to record videos must be the same. Below is a test image and its undistorted version that i used to test my pipeline.

![alt text][testCombo]

#### 2. Color transforms and gradients to find lane lines

In order to get lane lines from image I used the combination of following techniques

1. Region of interest - Only pixels that might have line information are considered. (block # 13)
2. White Lines from HLS Color space - Image by thresholding white range (block # 12)
3. Yellow Lines from HSV Color space - Image
again by thresholding yellow range (block # 12)

Below is extracted yellow and white lines and its gray-scaled version for further processing.

![alt text][comboLines]

#### 3. Perspective transform

Perspective transform is necessary in order to detect lines on planar surface. It is much easier to fit a polynomial from top of the image than by looking lines with an angle. Therefore, I had to find 4 points (source) that i can fit a polygon into, that has sides coinciding with lane lines. And another 4 points (destination) that I want these four images to be warped into, which is actually a perfect rectangular in 2D. These points are selected in Block # 18-19 as following:

``` python
src = np.float32(
    [[575, int(img_size[0] * 0.65)],
    [xDim -565, int(img_size[0] * 0.65)],
    [[207, img_size[0] - 1]],
    [[img_size[1] -180, img_size[0] - 1]])

dst = np.float32(
    [[455,0],
    [img_size[1]-455,0],
    [455, img_size[0]],
    [xDim-455, img_size[0]]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 575, 468      | 455, 0        |
| 715, 468      | 825, 0      |
| 207, 719     | 455, 720      |
| 1100, 719      | 825, 720        |


![alt text][roadFlat]

And here is warped lines, as if we are looking lines from a bird-eye. Now it is much easier to fit these lines in to a second degree polynomial.

![alt text][warpedBW]

#### 4. Fitting polynomial

Now since I have pixels that "mostly" corresponds to lane lines, I can fit a polynomial by using this points. However first we need to label the good pixels that I think that belongs to a line. Here is why we need to use sliding windows technique. I initialise the position of the first/most-lowest window by using histogram, which demonstrated where the most pixels are summed up. Here is the histogram of above shown B&W Image, where the white pixel values are summed up among y-axis in lower half of the image:

![alt text][hist]

Here can left and right lines clearly be seen and that is exactly how i used this information to position left and right windows as initialisation. Then the pixels located inside of this rectangular area are labeled as good pixel, which will be used as polynomial fitting.

Once I place the first left and right window, I slide current windows up and before labeling good pixel, I recenter the position of the window by taking the mean position of pixels inside of slided rectangular. Here a minimum value of 20 pixels are checked before recentering the window.    

This is done until all windows are slide until top of the image.

Now since we have the good pixel coordinates of left and right image, we can fit them into polynomial.

Additionally if no pixels are found for current line, I used same polynomial parameters from one last frame to visualise the current line:

``` python
# Fit a second order polynomial to each
global Glefty
global Gleftx
global Grighty
global Grightx

if lefty.size and leftx.size: # For Challange Videos Empty Polynomial Checker
    left_fit = np.polyfit(lefty, leftx, 2)
    Glefty = lefty
    Gleftx = leftx
else:
    left_fit = np.polyfit(Glefty, Gleftx, 2)

if righty.size and rightx.size:
    right_fit = np.polyfit(righty, rightx, 2)
    Grighty = righty
    Grightx = rightx
else:
    right_fit = np.polyfit(Grighty, Grightx, 2)
```

In order to calculate later the deviation of the car from centre, I took the average of the coordinates from left and right lines and fit it as a new center line. Below is the  visualisation of all lines by their colors:

1. Green: sliding windows
2. Red: good left pixels
3. Blue: good right pixels
4. Yellow: polynomial fits
5. Orange: center line found by the average of left and right line
6. Cyan: Image Center

![alt text][SlidingWin]
#### 5. Radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature is calculated by the polynomial parameters of the center line according to following formula:

![alt text][curvForm]

In order to get the curvature in meters I have assumed that the length of the line in y direction is 17 meters and the length between two lane line is 3.7 meters. Given centre line coordinates `calculateCurvature` function in block # 25 calculated the radius of curvature. For a smooth change between frames, I have summed up values of last 6 frames and took the averages of them in order to determine the radius. If the radius is found more that 2000 meters then it will be assumed that the car is going straight.

The deviation is calculated simply by finding the difference in x-axis between the center of the image and center line where y-axis is (img_size[1]/2, img_size[1]/2). Resulting pixel amount is then multiplied by meter per pixel value -> (3.7 / 900)

#### 6. Result
The pipeline in block # 29 is applied to all images to get a robust lane detection. Here is an example frame of the outcome.


![alt text][finalIm]

Additionally, I used double sided buffer `deque` to keep track of last 12 images and I used the averages of line positions in order to avoid jittering. This code can be found in block # 21


---

### Video

Here's a [link to my video result](https://youtu.be/uClw7qwSRn8)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Obviously the most challenging part of this project was detecting lines in changing lightning conditions for example shadows of the trees. In order to solve this problem I have implemented a color based threshold for yellow and white color. However, we can not expect this thresholds to work in all conditions. In challenge video, because of the shadow of bridge, tracking goes completely off for couple seconds with same parameters I used for project video.

In some frames there were no line found and algorithm failed to continue, therefore I have implemented a null-checker. When there is no line detected in any frame, then the line parameters from last frame will be taken.

Here, when we warp our image, we always assume that at that moment the road is flat. But this is not always the case as we can see in harder challenge video. Car might drive down- or up-hill and therefore warping & Region of Interest parameters should be adapted to the current condition.

In order to avoid jittering I have used lane information from last frames. That is however not very robust to strong curvature-radius changes as can be seen in harder challenge video. More real-time approach must be performed, when strong change in radius exists.


In conclusion, this is again a very naive way to detect lane lines. For a more reliable line detection, one needs more data (ex. GPS, slope-sensor, depth sensor etc.) than just a 2D image.

[//]: # (Image References)

[foundCor]: ./output_images/foundCor.png "foundCor"
[comboCalib2]: ./output_images/comboCalib2.png "comboCalib"
[testCombo]: ./output_images/comboTest.png "testCombo"
[comboLines]: ./output_images/comboLines.png "comboLines"
[roadFlat]: ./output_images/roadFlat.png "roadFlat"
[warpedBW]: ./output_images/warpedBW.png
[hist]: ./output_images/hist.png "hist"
[SlidingWin]: ./output_images/SlidingWin.png "SlidingWin"
[curvForm]: ./output_images/curvForm.png "SlidingWin"
[finalIm]: ./output_images/finalIm.png "finalIm"
