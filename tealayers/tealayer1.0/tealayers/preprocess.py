import cv2 
import numpy as np 

# img = cv2.imread("raw.jpg")
# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# ret, thresh = cv2.threshold(img,127,255,0)
# # cv2.imwrite("thresh.jpg",thresh)

# img_new = np.multiply(img,thresh)
# # print(img_new)
# img_new = cv2.equalizeHist(img_new)

# cv2.imwrite("new.jpg",img_new*1.5 + img*0.7)

def preprocess(img):
    img = cv2.equalizeHist(img)
    ret, thresh = cv2.threshold(img,127,255,0)
    img_new = np.multiply(img,thresh)
    img_new = cv2.equalizeHist(img_new)
    return img_new
