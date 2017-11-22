from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from hog import HOG
import dataset
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True)
ap.add_argument("-m", "--model", required=True)
args = vars(ap.parse_args())

(digits, target) = dataset.load_digits(args["dataset"])
data = []

hog = HOG(orientations=18, pixelsPerCell=(10, 10), cellsPerBlock=(1, 1), transform=True)

for image in digits:
    cv2.imshow("image", image)
    image = dataset.deskew(image, 20)
    cv2.imshow("deskew", image)
    image = dataset.center_extent(image, (20, 20))
    cv2.imshow("center_extent", image)
    # cv2.waitKey()

    hist = hog.describe(image)
    data.append(hist)

model = LinearSVC(random_state=42)
model.fit(data, target)

joblib.dump(model, args["model"])
