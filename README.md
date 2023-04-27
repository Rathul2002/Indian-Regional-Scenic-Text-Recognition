# Indian-Regional-Scenic-Text-Recognition

This approach shows how to detect regions in an image that contain text by using maximally stable extremal regions (MSER) feature detector. This is a common task performed on unstructured scenes. Unstructured scenes are images that contain undetermined or random scenarios. This is different than structured scenes, which contain known scenarios where the position of text is known beforehand.

Here, I have used MSER "Maximally Stable Extremal Regions" algorithm for text recognition and easyOCR for text Extraction.

What is MSER?

MSER stands for "Maximally Stable Extremal Regions" and refers to a computer vision algorithm used to detect regions or areas in an image that have consistent brightness or color.

To put it simply, MSER identifies and outlines regions of an image that have similar visual characteristics and are stable under different lighting conditions or image transformations. These regions can be used for various applications such as object recognition, tracking, and image segmentation.

The algorithm works by identifying pixels in an image that form connected components, and then analyzing the evolution of these components under different thresholds. Regions that remain stable across a range of thresholds are considered to be MSERs.

Overall, MSER is a powerful tool for computer vision tasks that require reliable and robust region detection in images.

MSER is present in Opencv library.


Required Library:
OpenCV for image processing and MSER.
Pytorch and easyocr for text extraction.
Matploitlib for image display.
