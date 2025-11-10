# **DIP-Midterm-Project**
This is my solutions for Midterm Project of 'Introduction to Digital Image Processing' Course. My code focuses on the color range of the traffic signs present in the given video (Blue, Red. Real life signs can consist of many other colors) using only Opencv and basic Python libraries (no machine learning) as required.

## **Task 1 (3.0 points):**
Given an input video (attached in the assignment): Draw rectangles surrounding each Traffic sign in all frames of the input video automatically, and save the outputs into a new video file.
## **Task 1 output results:**
![](https://github.com/VERISBABY/DIP-Midterm-Project/blob/79a30cf2797af383e183ffaac517750e842f3ee3/di1.png)
![](https://github.com/VERISBABY/DIP-Midterm-Project/blob/79a30cf2797af383e183ffaac517750e842f3ee3/di2.png)
![](https://github.com/VERISBABY/DIP-Midterm-Project/blob/79a30cf2797af383e183ffaac517750e842f3ee3/di3.png)

**Output video**: ![https://youtu.be/3FT55SsTC8E]

> Overall, the code solution can detected all major observable traffic signs of the given video. 
> However, there are still some objects not even being a traffic sign have been catched such as billborad,etc.
> The number of false positives is small so overall this coding solution solves the problem quite well.

### **Key Technical Attempt and Innovations**

- Multi-stage color segmentation uses multiple HSV ranges and morphological operations for robust color detection
- Geometric Property Analysis: Combines multiple shape descriptors for accurate classification
- I tried enhance contrast of each frame use Adaptive Histogram Equalization (CLAHE), cv2.equalizeHist but it not improved for given input video.
- Set up ‘area’ limitation is also improtant to reduce small wrong detection that not in standard traffic signs size.
- Multi-level Polygon Approximation: Handles shape detection at different scales
- Border-specific Detection: Specialized algorithm for hollow traffic signs
- Comprehensive Validation: Multiple verification steps to reduce false positives

## **Task 2 (4.0 points): Given an input image:**
Draw rectangles surrounding each digit in the input image automatically, and save the output 
image into a file.
## **Task 2 output result:**
![](https://github.com/VERISBABY/DIP-Midterm-Project/blob/00f4cef0f6ee68d1434c9cb359fc769f985c5438/523K0047.png)

### **Summary step by step i’ve used to solve the task:**
- Line removal: remove_lines
- Contrast & binarization: advanced_preprocessing (CLAHE + adaptive + Otsu)
- Morphological cleanup: smart_morphological_processing + remove_small_components
- Contour detection & filtering: contour_analysis
- Presentation & drawing: create_professional_frame
- Run end-to-end and return the final result: process and main

