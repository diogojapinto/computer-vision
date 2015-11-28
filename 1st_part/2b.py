import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt

img = np.full((50, 200, 3), 100, dtype='uint8')
print(img.shape)
img[25, 100, :] = 255

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
