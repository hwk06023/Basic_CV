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

## Geometric transform

### Translation 

### Rotation

### Affine transform

### Perspective transform

### Remapping



<br/><br/><br/><br/>


## Filltering

### Mean filter (blurring)
Mean filter is a simple filter that is usually used for blurring. <br/>
It is also called box filter. <br/>

```python
cv2.blur(src, ksize[, dst[, anchor[, borderType]]]) -> dst
```

[opencv docs](https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#ga8c45db9afe636703801b0b2e440fce37)

### Gaussian filter (blurring)
Gaussian filter is a filter that is usually used for blurring. <br/>
Different from mean filter, it is more effective for blurring. <br/>

```python
cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) -> dst
```

[opencv docs](https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1)

### Unsharp masking (sharpening)

```python
cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]]) -> dst
```

[opencv docs](https://docs.opencv.org/3.4/d5/dc4/tutorial_adding_images.html)

### Laplacian filter (sharpening)

```python
cv2.Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]]) -> dst
```

[opencv docs](https://docs.opencv.org/3.4/d5/db5/tutorial_laplace_operator.html)

### Median filter (noise removal)

```python
cv2.medianBlur(src, ksize[, dst]) -> dst
```

[opencv docs](https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#ga564869aa33e58769b4469101aac458f9)

### Bilateral filter (noise removal)

```python
cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst[, borderType]]) -> dst
```

[opencv docs](https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed)



### Image Pyramids

Useful for image blending, image resizing, image compression, image reconstruction

[opencv docs](https://docs.opencv.org/3.4/dc/dff/tutorial_py_pyramids.html)



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

#### SIFT (Scale-Invariant Feature Transform)

1. **Scale-space extrema detection**: The first stage of computation searches over all scales and image locations. It is implemented efficiently by using a difference-of-Gaussian function to identify potential interest points that are invariant to scale and orientation.

2. **Keypoint localization**: At each candidate location, a detailed model is fit to determine location and scale. Keypoints are selected based on measures of their stability.

3. **Orientation assignment**: One or more orientations are assigned to each keypoint lo- cation based on local image gradient directions. All future operations are performed on image data that has been transformed relative to the assigned orientation, scale, and location for each feature, thereby providing invariance to these transformations.

4. **Keypoint descriptor**: The local image gradients are measured at the selected scale in the region around each keypoint. These are transformed into a representation that allows for significant levels of local shape distortion and change in illumination.

<br/>

```python
cv2.SIFT_create([, nfeatures[, nOctaveLayers[, contrastThreshold[, edgeThreshold[, sigma]]]]]) -> retval
```

[opencv docs 1](https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html)

[opencv docs 2](https://docs.opencv.org/3.4/d7/d60/classcv_1_1SIFT.html)

#### SURF (Speeded-Up Robust Features)

```python
cv2.SURF_create([, hessianThreshold[, nOctaves[, nOctaveLayers[, extended[, upright]]]]]) -> retval
```

[opencv docs](https://docs.opencv.org/3.4/df/dd2/tutorial_py_surf_intro.html)

#### ORB (Oriented FAST and Rotated BRIEF)

```python
cv2.ORB_create([, nfeatures[, scaleFactor[, nlevels[, edgeThreshold[, firstLevel[, WTA_K[, scoreType[, patchSize[, fastThreshold]]]]]]]]]) -> retval
```

[opencv docs](https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html)

#### BRISK (Binary Robust Invariant Scalable Keypoints)

```python
cv2.BRISK_create([, thresh[, octaves[, patternScale]]]) -> retval
```

[opencv docs](https://docs.opencv.org/3.4/d8/d30/classcv_1_1BRISK.html)

#### BRIEF (Binary Robust Independent Elementary Features)

```python
cv2.BriefDescriptorExtractor_create([, bytes[, use_orientation]]) -> retval
```

[opencv docs](https://docs.opencv.org/3.4/dc/d7d/classcv_1_1xfeatures2d_1_1BriefDescriptorExtractor.html)

#### FREAK (Fast Retina Keypoint)

```python
cv2.FREAK_create([, orientationNormalized[, scaleNormalized[, patternScale[, nOctaves[, selectedPairs]]]]]) -> retval
```

[opencv docs](https://docs.opencv.org/3.4/df/db4/classcv_1_1xfeatures2d_1_1FREAK.html)



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











