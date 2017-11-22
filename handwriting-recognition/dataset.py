import numpy as np
import imutils
import mahotas
import cv2

def load_digits(datasetPath):
    data = np.genfromtxt(datasetPath, delimiter=",", dtype="uint8")
    target = data[:, 0]
    data = data[:, 1:].reshape(data.shape[0], 28, 28)
    print("data=%s" % str(data.shape))

    return data, target

def deskew(image, width):
    (h, w) = image.shape[:2]
    print("origin image.shape=%s" % str(image.shape))
    m = cv2.moments(image)
    if abs(m['mu02']) < 1e-2:
        return image.copy()
    skew = m["mu11"] / m["mu02"]
    M = np.float32([[1, skew, -0.5*w*skew], [0, 1, 0]])
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    image = imutils.resize(image, width=width)
    print("new image.shape=%s" % str(image.shape))
    return image

def center_extent(image, size):
    (eW, eH) = size
    print("eW, eH = %s" % str(size))
    print("image.shape=%s" % str(image.shape))

    if image.shape[1] > image.shape[0]:
        image = imutils.resize(image, width=eW)
    else:
        image = imutils.resize(image, height=eH)

    print("resize image.shape=%s" % str(image.shape))
    extent = np.zeros((eH, eW), dtype="uint8")
    offsetX = (eW - image.shape[1]) // 2
    offsetY = (eH - image.shape[0]) // 2
    extent[offsetY:offsetY + image.shape[0], offsetX:offsetX + image.shape[1]] = image
    print("offsetX, offsetY = (%s, %s)" % (offsetX, offsetY))
    cv2.imshow("extent", extent)

    CM = mahotas.center_of_mass(extent)
    (cY, cX) = np.round(CM).astype("int32")
    (dX, dY) = ((size[0] // 2) - cX, (size[1] // 2) - cY)
    M = np.float32([[1, 0, dX], [0, 1, dY]])
    extent = cv2.warpAffine(extent, M, size)
    print("CM=%s" % CM)
    print("(cY, cX) = (%s, %s)" % (cY, cX))
    print("(dX, dY) = (%s, %s)" % (dX, dY))
    print("M=%s" % M)
    cv2.imshow("extent1", extent)
    return extent
