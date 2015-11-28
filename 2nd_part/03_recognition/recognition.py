import cv2
import numpy as np
from matplotlib import pyplot as plt

NR_POSTERS = 7

### FUNCTIONS ###

def import_image(path):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return (img, img_gray)

### MAIN ###

test_poster, test_poster_gray = import_image('../02_local_features/posters/poster_test.jpg')
posters = [import_image("../02_local_features/posters/poster{}.jpg".format(i))
           for i in range(1, NR_POSTERS + 1)]

# create SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# extract descriptors in test_poster
test_poster_kps, test_poster_descs = sift.detectAndCompute(test_poster_gray, None)

# extract descriptors in posters
posters_descs = [sift.detectAndCompute(x[1], None) for x in posters]

all_descs = [x[1] for x in posters_descs]
all_descs = [item for sublist in all_descs for item in sublist]
all_descs = np.asarray(all_descs)

# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Apply KMeans
compactness, labels, centers = cv2.kmeans(all_descs, 10, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

#test_poster_show = cv2.drawKeypoints(test_poster, kps, None)
#test_poster_show = cv2.cvtColor(test_poster_show, cv2.COLOR_BGR2RGB)
