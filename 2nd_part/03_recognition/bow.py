import cv2
import numpy as np
from matplotlib import pyplot as plt
from itertools import groupby

NR_POSTERS = 7
NR_GROUPS = 10

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

# BOW training

bow_trainer = cv2.BOWKMeansTrainer(100, attempts=1, flags=cv2.KMEANS_PP_CENTERS) # cv2.TermCriteria_MAX_ITER,
bow_extractor = cv2.BOWImgDescriptorExtractor(bow_trainer, matcher)
