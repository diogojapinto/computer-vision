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

# create FLANN matcher
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

# extract descriptors in test_poster
test_poster_kps, test_poster_descs = sift.detectAndCompute(test_poster_gray, None)

# extract descriptors in posters
posters_descs = [sift.detectAndCompute(x[1], None) for x in posters]

other_descs = [x[1] for x in posters_descs]
all_descs = [item for sublist in other_descs for item in sublist]
all_descs = np.asarray(all_descs)

# BOW training

bow_trainer = cv2.BOWKMeansTrainer(100, attempts=1, flags=cv2.KMEANS_PP_CENTERS) # cv2.TermCriteria_MAX_ITER,
bow_extractor = cv2.BOWImgDescriptorExtractor(sift, flann)

bow_trainer.add(all_descs)
vocabulary = bow_trainer.cluster()

bow_extractor.setVocabulary(vocabulary)

test_bow_des = bow_extractor.compute(test_poster, test_poster_kps)

other_bow_des = [bow_extractor.compute(poster[1], desc[0]) \
                 for poster, desc in zip(posters, posters_descs)]

#print([method for method in dir(bow_trainer) if callable(getattr(bow_trainer, method))])

scored_posters = []

for i, bow_des in enumerate(other_bow_des):
    test_bow = np.asarray(test_bow_des)
    other_bow = np.asarray(bow_des)

    score = test_bow.dot(other_bow.transpose())[0][0]

    scored_posters.append((score, i))

scored_posters = sorted(scored_posters, reverse=True)

for score, i in scored_posters:
    print("Poster {} scored {}".format(i+1, score))
