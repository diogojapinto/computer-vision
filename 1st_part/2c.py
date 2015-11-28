import cv2
import numpy as np

class Image:

    def __init__(self, img):
        self.img = img

    @classmethod
    def fromFile(cls, filepath):
        img = cv2.imread(filepath)
        return cls(img)

    @classmethod
    def fromSpecs(cls, height, width, intensity):
        img = np.full((height, width, 3), intensity, dtype='uint8')
        return cls(img)

img1 = Image.fromFile('sigarra.jpg')
img2 = Image.fromSpecs(50, 200, 100)

cv2.imshow('from file', img1.img)
cv2.imshow('from specs', img2.img)

cv2.waitKey(0)
cv2.destroyAllWindows()
