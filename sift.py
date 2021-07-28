import numpy as np
import cv2
from matplotlib import pyplot as plt


print("Loading images", flush=True)
img1 = cv2.imread('Image4.jpg', 0)          # queryImage
img2 = cv2.imread('nust.tif', 0)            # trainImage

# Initiate SIFT detector
print("Loading SIFT", flush=True)
orb = cv2.ORB()

# find the keypoints and descriptors with SIFT
print("Loading keypoints", flush=True)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Match descriptors.
print("Finding matches", flush=True)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
print("Sorting matches", flush=True)
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches.
print("Drawing matches", flush=True)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], flags=2)
plt.imshow(img3),plt.show()
