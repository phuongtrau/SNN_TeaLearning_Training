import cv2 
import numpy as np 

# img = cv2.imread("img_raw_after.jpg")


# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray)

# ret, thresh = cv2.threshold(img,200,255,1)

# img_blur = cv2.GaussianBlur(,(7,7),0)

# img_lap = cv2.Laplacian(img_blur,cv2.cv2.CV_64F)

# new_thresh = np.multiply(thresh,img_lap)

# thresh_1 = np.array(new_thresh!=0).astype(float)
# kernel = np.ones((3,3),np.uint8)

# erosion = cv2.dilate(thresh,kernel,5)
# cv2.imwrite("ero.jpg",erosion)
# invert_erosion = 1- erosion

# new_img = np.multiply(img,invert_erosion)

# cv2.imwrite("new.jpg",clahe)

# def preprocess(img):
#     img = cv2.equalizeHist(img)
#     ret, thresh = cv2.threshold(img,127,255,0)
#     img_new = np.multiply(img,thresh)
#     img_new = cv2.equalizeHist(img_new)
#     return img_new
