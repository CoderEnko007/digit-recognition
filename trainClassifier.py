from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn import datasets
from skimage.feature import hog
import numpy as np

dataset = datasets.fetch_mldata("MNIST Original")
features = np.array(dataset.data, "int16")
labels = np.array(dataset.target, "int")

list_hog_feature = []
for feature in features:
    hf = hog(feature.reshape(28, 28), orientations=9, pixels_per_cell=(14, 14),
             cells_per_block=(1, 1), visualise=False)
    list_hog_feature.append(hf)
hog_features = np.array(list_hog_feature, "float64")

clf = LinearSVC()
clf.fit(hog_features, labels)
joblib.dump(clf, "digit_hog.pkl", compress=3)
