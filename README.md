# Basic_CV

This Repository is for basic computer vision(Detection, Recognition, Tracking). <br/>
Honestly, This Repository is for reference when I develop something (about CV) <br/>

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

```python
cv2.add(src1, src2[, dst[, mask[, dtype]]]) -> dst
```

[opencv docs](https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga10ac1bfb180e2cfda1701d06c24fdbd6)

```python
cv2.subtract(src1, src2[, dst[, mask[, dtype]]]) -> dst
```

[opencv docs](https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga10ac1bfb180e2cfda1701d06c24fdbd6)

```python
cv2.multiply(src1, src2[, dst[, scale[, dtype]]]) -> dst
```

[opencv docs](https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga10ac1bfb180e2cfda1701d06c24fdbd6)

```python
cv2.divide(src1, src2[, dst[, scale[, dtype]]]) -> dst
```

[opencv docs](https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga10ac1bfb180e2cfda1701d06c24fdbd6)

<br/>

### Histogram

```python
cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]) -> hist
```

[opencv docs](https://docs.opencv.org/3.4/d6/dc7/group__imgproc__hist.html#ga4b2b5fd75503ff9e6844cc4dcdaed35d)

```python
cv2.equalizeHist(src[, dst]) -> dst
```

[opencv docs](https://docs.opencv.org/3.4/d5/daf/tutorial_py_histogram_equalization.html)

```python
cv2.compareHist(H1, H2, method) -> retval
```

[opencv docs](https://docs.opencv.org/3.4/d6/dc7/group__imgproc__hist.html#ga994f53817d621e2e4228fc646342d386)


<br/><br/><br/><br/>

## Geometric transform

### Translation 

```python
cv2.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) -> dst
```

[opencv docs](https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983)

### Rotation

```python
cv2.getRotationMatrix2D(center, angle, scale) -> retval
```

[opencv docs](https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983)

### Affine transform

```python
cv2.getAffineTransform(src, dst) -> retval
```

[opencv docs](https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga8f6d378f9f8eebb5cb55cd3ae295a999)


### Perspective transform

```python
cv2.getPerspectiveTransform(src, dst[, solveMethod]) -> retval
```

[opencv docs](https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga8f6d378f9f8eebb5cb55cd3ae295a999)


### Remapping

```python
cv2.remap(src, map1, map2, interpolation[, dst[, borderMode[, borderValue]]]) -> dst
```

[opencv docs](https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121)



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

```python
cv2.pyrUp(src[, dst[, dstsize[, borderType]]]) -> dst
```

```python
cv2.pyrDown(src[, dst[, dstsize[, borderType]]]) -> dst
```




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

1. Compute x and y derivatives of image

2. Compute magnitude of gradient at every pixel

3. Eliminate pixels that are not local maxima of gradient magnitude

4. Hysteresis thresholding
- Select the pixels such That the gradient magnitude is larger than a high threshold

- Select the pixels such that the gradient magnitude is larger than a low threshold and that are connected to high threshold pixels

<br/>

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

```python
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount[, mode]) -> mask, bgdModel, fgdModel
```

[opencv docs](https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html)


### Moments

```python   
cv2.moments(array[, binaryImage]) -> retval
```

[opencv docs](https://docs.opencv.org/3.4/d8/d23/classcv_1_1Moments.html)


### Template matching

```python
cv2.matchTemplate(image, templ, method[, result[, mask]]) -> result
```

[opencv docs](https://docs.opencv.org/3.4/df/dfb/group__imgproc__object.html#ga586ebfb0a7fb604b35a23d85391329be)


### Cascade classifier

```python
cv2.CascadeClassifier([filename]) -> <CascadeClassifier object>
```

[opencv docs](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)


### HOG (Histogram of Oriented Gradients)

```python
cv2.HOGDescriptor([_winSize[, _blockSize[, _blockStride[, _cellSize[, _nbins[, _derivAperture[, _winSigma[, _histogramNormType[, _L2HysThreshold[, _gammaCorrection[, _nlevels[, _signedGradient]]]]]]]]]]]]) -> <HOGDescriptor object>
```

[opencv docs](https://docs.opencv.org/3.4/d5/d33/structcv_1_1HOGDescriptor.html)


<br/><br/><br/><br/>


## feature-point(keypoints) Detect and match

### Harris corner detection



```python
cv2.cornerHarris(src, blockSize, ksize, k[, dst[, borderType]]) -> dst
```

[opencv docs](https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html)

### Shi-Tomasi corner detection

```python
cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance[, corners[, mask[, blockSize[, useHarrisDetector[, k]]]]]) -> corners
```

[opencv docs](https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541)

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

<br/>

### Matcher(Brute-Force, FLANN)

#### Brute-Force

```python 
cv2.BFMatcher([, normType[, crossCheck]]) -> <BFMatcher object>
```

[opencv docs](https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html)


#### FLANN (Fast Library for Approximate Nearest Neighbors)

```python
cv2.FlannBasedMatcher([, indexParams[, searchParams]]) -> <FlannBasedMatcher object>
```

[opencv docs](https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html)


<br/><br/><br/><br/>



## Object tracking and Motion vector

### BackgroundSubtraction

#### MOG1, MOG2 (Mixture of Gaussian)
Gaussian Mixture-based Background/Foreground Segmentation Algorithm <br/>

```python
cv.bgsegm.createBackgroundSubtractorMOG([, history[, nmixtures[, backgroundRatio[, noiseSigma]]]]) -> retval
```

[opencv docs](https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html)

```python
cv.createBackgroundSubtractorMOG2([, history[, varThreshold[, detectShadows]]]) -> retval
```

[opencv docs](https://docs.opencv.org/3.4/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html)

#### GMG (Gaussian Mixture-based Background/Foreground Segmentation Algorithm)

```python 
cv.bgsegm.createBackgroundSubtractorGMG([, initializationFrames[, decisionThreshold]]) -> retval
```

[opencv docs](https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html)


### Moving Average

```python
cv2.accumulate(src, dst[, mask]) -> dst
```

[opencv docs](https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga8c1d2e21b9e1e3dd348488990849b1ed)

```python
cv2.accumulateWeighted(src, dst, alpha[, mask]) -> dst
```

[opencv docs](https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga8c1d2e21b9e1e3dd348488990849b1ed)



### MeanShift

```python
cv2.meanShift(probImage, window, criteria) -> retval, window
```

[opencv docs](https://docs.opencv.org/3.4/db/df8/tutorial_py_meanshift.html)

### CamShift   

```python
cv2.CamShift(probImage, window, criteria) -> retval, window
```

[opencv docs](https://docs.opencv.org/3.4/db/df8/tutorial_py_meanshift.html)


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











