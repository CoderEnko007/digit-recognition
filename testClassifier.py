from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)
args = vars(ap.parse_args())

clf = joblib.load("digit_hog.pkl")
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY_INV)[-1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

rects = [cv2.boundingRect(c) for c in cnts]
cv2.imshow("image", image)
for c in cnts:
    rect = cv2.boundingRect(c)
    (x, y, w, h) = rect
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
    # cv2.imshow("image", image)
    # leng = int(h*1.6)
    # pt1 = int(y + h) // 2 - leng // 2
    # pt2 = int(x + w) // 2 - leng // 2
    # roi = thresh[pt1:pt1+leng, pt2:pt2+leng]
    roi = thresh[y:y+h, x:x+w]
    cv2.imshow("roi", roi)

    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    roi_hog = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    nbr = clf.predict(np.array([roi_hog]))
    cv2.putText(image, str(int(nbr[0])), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
    cv2.imshow("image", image)
    print(clf.decision_function(np.array([roi_hog])))
    cv2.waitKey()
