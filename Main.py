from skimage.measure import compare_ssim
import imutils
import cv2
import numpy as np

# Define our color ranges
pinkLower = np.array([0,80,80], dtype=np.uint8)
pinkUpper = np.array([20,255,255], dtype=np.uint8)
pinkLower2 = np.array([135,55,55], dtype=np.uint8)
pinkUpper2 = np.array([180,255,255], dtype=np.uint8)
whiteLower = np.array([0,0,180], dtype=np.uint8)
whiteUpper = np.array([180,80,255], dtype=np.uint8)

MAX_FEATURES = 100000
GOOD_MATCH_PERCENT = 0.15


def alignImages(im1, im2):

  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)

  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]

  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)

  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))

  return im1Reg, h


def Create_Mask(mask,lower,upper):
    mask = cv2.inRange(mask,lower,upper)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)
    return mask

# Reads our images and resizes them
imageA = cv2.imread('Branch1.jpg')
imageA = imutils.resize(imageA, width = 300)
imageB = cv2.imread('Branch2.jpg')
imageB = imutils.resize(imageB, width = 300)

imageA,h = alignImages(imageA,imageB)
imageB,h = alignImages(imageB,imageA)

# Converts our images to HSV
HSVA = cv2.cvtColor(imageA, cv2.COLOR_BGR2HSV)
HSVB = cv2.cvtColor(imageB, cv2.COLOR_BGR2HSV)

# Creates our masks for our first image
pinkA = Create_Mask(HSVA,pinkLower,pinkUpper)
pinkA2 = Create_Mask(HSVA,pinkLower2,pinkUpper2)
whiteA = Create_Mask(HSVA,whiteLower,whiteUpper)

# Creates our masks for our second image
pinkB = Create_Mask(HSVB,pinkLower,pinkUpper)
pinkB2 = Create_Mask(HSVB,pinkLower2,pinkUpper2)
whiteB = Create_Mask(HSVB,whiteLower,whiteUpper)

# Combines our masks
FilteredA =  pinkA2 | whiteA
FilteredAImage = cv2.bitwise_and(imageA,imageA,mask=FilteredA)
FilteredB =  pinkB2 | whiteB
FilteredBImage = cv2.bitwise_and(imageB,imageB,mask=FilteredB)

# Compares our masked images
(score, diff) = compare_ssim(FilteredA, FilteredB, full = True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

# Thresholds our difference image
r, thresh = cv2.threshold(diff,0,255,cv2.THRESH_BINARY_INV)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for c in cnts:
    area = cv2.contourArea(c)

    if (area > 100):
        (x, y, w, h,) = cv2.boundingRect(c)
        cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)


#cv2.imshow("1", FilteredA)
cv2.imshow("imageA",imageA)
#cv2.imshow("2", FilteredB)
cv2.imshow("imageB",imageB)
cv2.imshow("diff", diff)
cv2.imshow("thresh", thresh)
cv2.waitKey(0)
