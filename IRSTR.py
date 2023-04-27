import numpy as np
import cv2
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import easyocr

# **************************************Reading image and preprocess**********************************************
imgname = "./examples/test4.jpg"
img = cv2.imread(imgname)
plt.figure(figsize=(5,5))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 3)

# ******************************************Getting MSER regions***************************************************
mser = cv2.MSER_create()
mser.setMinArea(100)
mser.setMaxArea(7000)
mser.setDelta(4)
coordinates, bboxes = mser.detectRegions(gray)

# *****************************Filtering the regions and filling with random colours*******************************
vis = img.copy()
coords = []
for coord in coordinates:
    bbox = cv2.boundingRect(coord)
    x, y, w, h = bbox
    aspect_ratio = float(w) / h
    if w < 10 or h < 10 or aspect_ratio > 5 or aspect_ratio < 0.2:
        continue
    coords.append(coord)

colors = [[43, 43, 200], [43, 75, 200], [43, 106, 200], [43, 137, 200], [43, 169, 200], [43, 200, 195], [43, 200, 163], [43, 200, 132], [43, 200, 101], [43, 200, 69], [54, 200, 43], [85, 200, 43], [116, 200, 43], [148, 200, 43], [179, 200, 43], [200, 184, 43], [200, 153, 43], [200, 122, 43], [200, 90, 43], [200, 59, 43], [200, 43, 64], [200, 43, 95], [200, 43, 127], [200, 43, 158], [200, 43, 190], [174, 43, 200], [142, 43, 200], [111, 43, 200], [80, 43, 200], [43, 43, 200]]
np.random.seed(0)
final_img = np.ones_like(img) *255
for cnt in coords:
    xx = cnt[:, 0]
    yy = cnt[:, 1]
    color = colors[np.random.choice(len(colors))]
    #color=[0,0,0]
    final_img[yy, xx] = color

#**************************************Sharpening the image for better Detection**********************************
image_data = np.array(final_img)
image = Image.fromarray(image_data)
sharp_image = image.filter(ImageFilter.SHARPEN)
sharp_image_data = np.array(sharp_image)
final_img=sharp_image_data
plt.figure(figsize=(5,5))
plt.imshow(final_img)
plt.title("Image after Recognition")
plt.show()

#***************************************Using EasyOCR for text extraction****************************************** 
reader = easyocr.Reader(['hi','en'], gpu=True)
result = reader.readtext(final_img)
length = len(result)
text=''
for i in range(length):
    text=text+' '+result[i][1]

print("Text Detected from image are:")
print(text)