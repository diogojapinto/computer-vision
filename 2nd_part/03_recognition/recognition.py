import cv2
import numpy as np
from matplotlib import pyplot as plt
from itertools import groupby
from math import log

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

# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, NR_GROUPS, 1.0)

# Apply KMeans
compactness, labels, centers = cv2.kmeans(all_descs, NR_GROUPS, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# analyse the amount of descriptors distributed by cluster
labels_flat = [item for sublist in labels for item in sublist]
plt.hist(labels_flat, NR_GROUPS - 1)
plt.show()

# analyze the proximity of poster_test to the vocabulary
def whichWord(desc, vocabulary):
    min_dist = float('inf')
    min_word = -1

    for word, w_desc in enumerate(vocabulary):
        dist = cv2.norm(desc, w_desc, cv2.NORM_L2)

        min_dist, min_word = \
            (dist, word) if dist < min_dist else (min_dist, min_word)

    return min_word


test_poster_words = [whichWord(desc, centers) for desc in test_poster_descs]

# draw keypoints collored according to the word they correspond to
def draw_words(img, words, nr_words, kps):

    img_ret = img.copy()
    step = int((255 / (log(nr_words + 1) / log(3))))
    colors = [(b, g, r) for b in range(0, 256, step) \
                        for g in range(0, 256, step) \
                        for r in range(0, 256, step)
             ]

    for i, (kp, word) in enumerate(zip(kps, words)):
        img_ret = cv2.drawKeypoints(img_ret, [kp], None, color=colors[word])

    return img_ret

test_poster_show = draw_words(test_poster,
                              test_poster_words,
                              NR_GROUPS,
                              test_poster_kps)

other_posters_show = []
for i in range(NR_POSTERS):
    other_poster_words = [whichWord(desc, centers) \
                          for desc in posters_descs[i][1]]

    other_poster_show = draw_words(posters[i][0],
                                   other_poster_words,
                                   NR_GROUPS,
                                   posters_descs[i][0])

    other_posters_show.append(other_poster_show)

# show it all
test_poster_show = cv2.cvtColor(test_poster_show, cv2.COLOR_BGR2RGB)

for i in range(NR_POSTERS):
    other_poster_show = cv2.cvtColor(other_posters_show[i], cv2.COLOR_BGR2RGB)

    plt.subplot(121); plt.imshow(test_poster_show);
    plt.subplot(122); plt.imshow(other_poster_show);
    plt.tight_layout(); plt.show()
