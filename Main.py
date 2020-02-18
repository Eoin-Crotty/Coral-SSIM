from skimage.measure import compare_ssim
import imutils
import cv2
import numpy as np

# Define our color ranges
pinkLower = np.array([0,80,80], dtype=np.uint8)
pinkUpper = np.array([20,255,255], dtype=np.uint8)
pinkLower2 = np.array([140,70,70], dtype=np.uint8)
pinkUpper2 = np.array([180,255,255], dtype=np.uint8)
whiteLower = np.array([0,0,200], dtype=np.uint8)
whiteUpper = np.array([180,20,255], dtype=np.uint8)


def Create_Mask(mask,lower,upper):
    mask = cv2.inRange(mask,lower,upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask

# Reads our images
imageA = cv2.imread('Branch1.jpg')
imageA = imutils.resize(imageA, width = 300)
imageB = cv2.imread('Branch2.jpg')
imageB = imutils.resize(imageB, width = 300)

HSVA = cv2.cvtColor(imageA, cv2.COLOR_BGR2HSV)
HSVB = cv2.cvtColor(imageB, cv2.COLOR_BGR2HSV)

pinkA = Create_Mask(HSVA,pinkLower,pinkUpper)
pinkA2 = Create_Mask(HSVA,pinkLower2,pinkUpper2)
whiteA = Create_Mask(HSVA,whiteLower,whiteUpper)

pinkB = Create_Mask(HSVB,pinkLower,pinkUpper)
pinkB2 = Create_Mask(HSVB,pinkLower2,pinkUpper2)
whiteB = Create_Mask(HSVB,whiteLower,whiteUpper)

FilteredA =  pinkA2 | whiteA
FilteredAImage = cv2.bitwise_and(imageA,imageA,mask=FilteredA)
FilteredB =  pinkB2 | whiteB
FilteredBImage = cv2.bitwise_and(imageB,imageB,mask=FilteredB)


(score, diff) = compare_ssim(FilteredA, FilteredB, full = True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))
'''
thresh = cv2.threshold(diff, 0, 255,
         cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findCountours(thresh.copy(), cv2.RETR_EXTERNAL,
       cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for c in cnts:
    (x, y, w, h,) = cv2.boundingRect(c)
    cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
'''
cv2.imshow("1", FilteredA)
cv2.imshow("imageA",imageA)
cv2.imshow("2", FilteredB)
cv2.imshow("imageB",imageB)
cv2.imshow("diff", diff)
cv2.waitKey(0)
