# Basic_CV

This Repository is for basic computer vision(Detection, Recognition, Tracking).

## OUTLINE

- [Basic process](#basic-process)
  * [point processing for brightness and contrast](#point-processing-for-brightness-and-contrast)
  * [Histogram](#histogram)
- [Filltering](#filltering)
  * [mean filter (blurring)](#mean-filter--blurring-)
  * [Gaussian filter (blurring)](#gaussian-filter--blurring-)
  * [Unsharp masking (sharpening)](#unsharp-masking--sharpening-)
  * [Laplacian filter (sharpening)](#laplacian-filter--sharpening-)
  * [Median filter (noise removal)](#median-filter--noise-removal-)
  * [Bilateral filter (noise removal)](#bilateral-filter--noise-removal-)
  * [Image Pyramids](#image-pyramids)
- [Geometric transform](#geometric-transform)
  * [Translation](#translation)
  * [Rotation](#rotation)
  * [Affine transform](#affine-transform)
  * [Perspective transform](#perspective-transform)
  * [Remapping](#remapping)
- [Extract feature](#extract-feature)
  * [Edge detection & Derivative (Sobel, Laplacian, Canny)](#edge-detection---derivative--sobel--laplacian--canny-)
  * [Hough transform](#hough-transform)
- [ImageSegmentation ObjectDetection](#imagesegmentation-objectdetection)
  * [GrabCut](#grabcut)
  * [Moments](#moments)
  * [Template matching](#template-matching)
  * [Cascade classifier](#cascade-classifier)
  * [HOG (Histogram of Oriented Gradients)](#hog--histogram-of-oriented-gradients-)
- [feature-point(keypoints) Detect and match](#feature-point-keypoints--detect-and-match)
  * [Harris corner detection](#harris-corner-detection)
  * [Shi-Tomasi corner detection](#shi-tomasi-corner-detection)
  * [Descriptor (SIFT, SURF, ORB, BRISK, BRIEF, FREAK ..)](#descriptor--sift--surf--orb--brisk--brief--freak--)
  * [FLANN (Fast Library for Approximate Nearest Neighbors)](#flann--fast-library-for-approximate-nearest-neighbors-)
- [Object tracking and Motion vector](#object-tracking-and-motion-vector)
  * [BackgroundSubtraction](#backgroundsubtraction)
  * [Moving Average](#moving-average)
  * [MOG(Mixture of Gaussian)](#mog-mixture-of-gaussian-)
  * [MeanShift](#meanshift)
  * [CamShift](#camshift)
  * [OpticalFlow](#opticalflow)
  * [Lucas-Kanade](#lucas-kanade)
  * [Dense Optical Flow](#dense-optical-flow)
- [(+) Binarization](#----binarization)
  * [Otsu's Binarization](#otsu-s-binarization)
  * [Adaptive Thresholding](#adaptive-thresholding)
  * [Watershed transform](#watershed-transform)
  * [Morphology](#morphology)
  * [Connected Component Labeling](#connected-component-labeling)
  * [Contour detection](#contour-detection)
  * [Convex Hull](#convex-hull)


## Basic process

### Point processing for brightness and contrast

### Histogram





<br/><br/><br/><br/>

## Filltering

### Mean filter (blurring)

### Gaussian filter (blurring)

### Unsharp masking (sharpening)

### Laplacian filter (sharpening)

### Median filter (noise removal)

### Bilateral filter (noise removal)

### Image Pyramids


<br/><br/><br/><br/>


## Geometric transform

### Translation 

### Rotation

### Affine transform

### Perspective transform

### Remapping


<br/><br/><br/><br/>


## Extract feature

### Edge detection & Derivative (Sobel, Laplacian, Canny)

#### Sobel

```python
cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]) -> dst
```

[opencv docs](https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html)

#### Laplacian

```python
cv2.Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]]) -> dst
```

[opencv docs](https://docs.opencv.org/3.4/d5/db5/tutorial_laplace_operator.html)

#### Canny

```python
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]) -> edges
```

[opencv docs](https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html)

### Hough transform

#### Hough line transform

```python
cv2.HoughLines(image, rho, theta, threshold[, lines[, srn[, stn[, min_theta[, max_theta]]]]]) -> lines
```

[opencv docs](https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html)

#### Hough circle transform

```python
cv2.HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
```

[opencv docs](https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html)




<br/><br/><br/><br/>


## ImageSegmentation ObjectDetection

### GrabCut

### Moments

### Template matching

### Cascade classifier

### HOG (Histogram of Oriented Gradients)





<br/><br/><br/><br/>


## feature-point(keypoints) Detect and match


### Harris corner detection


### Shi-Tomasi corner detection


### Descriptor (SIFT, SURF, ORB, BRISK, BRIEF, FREAK ..)

### FLANN (Fast Library for Approximate Nearest Neighbors)


<br/><br/><br/><br/>



## Object tracking and Motion vector

### BackgroundSubtraction

### Moving Average

### MOG(Mixture of Gaussian) 
Gaussian Mixture-based Background/Foreground Segmentation Algorithm <br/>



### MeanShift

### CamShift   


### Lucas-Kanade


### OpticalFlow

### Dense Optical Flow





<br/><br/><br/><br/>

## (+) Binarization

### Otsu's Binarization

### Adaptive thresholding

### Watershed transform

### Morphology

#### Erosion

#### Dilation

#### Opening

#### Closing

### Connected Component Labeling

### Contour detection

### Convex Hull











