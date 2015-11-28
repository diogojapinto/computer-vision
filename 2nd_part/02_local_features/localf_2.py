import cv2
from matplotlib import pyplot as plt

NR_POSTERS = 7

### FUNCTIONS ###

def import_image(path):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return (img, img_gray)


### MAIN ###

test_poster, test_poster_gray = import_image('./posters/poster_test.jpg')
posters = [import_image("./posters/poster{}.jpg".format(i))
           for i in range(1, NR_POSTERS + 1)]



# create SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
#surf = cv2.xfeatures2d.SURF_create()

# create FLANN matcher
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

# extract descriptors in test_poster
test_poster_kps, test_poster_descs = sift.detectAndCompute(test_poster_gray, None)

# extract descriptors in posters
posters_descs = [sift.detectAndCompute(x[1], None) for x in posters]

# match descriptors
matches = [flann.knnMatch(test_poster_descs, x[1], k=2) for x in posters_descs]
matchesMasks = [[[0,0] for _ in range(len(matches[i]))] for i in range(len(matches))]

draw_params = [dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMasks[i],
                   flags = 0) for i in range(len(matchesMasks))]

# ratio test as per Lowe's paper
for i in range(len(matches)):
    for j, (m, n) in enumerate(matches[i]):
        if m.distance < 0.7 * n.distance:
            matchesMasks[i][j] = [1,0]

matches_posters_no_filtering = [cv2.drawMatchesKnn(test_poster, test_poster_kps, posters[i][0], posters_descs[i][0], matches[i], None, **(draw_params[i])) for i in range(NR_POSTERS)]

#test_poster_show = cv2.drawKeypoints(test_poster, kps, None)
#test_poster_show = cv2.cvtColor(test_poster_show, cv2.COLOR_BGR2RGB)

# plot without filtering
plt.gcf().canvas.set_window_title('Without Filtering')
for i in range(NR_POSTERS):
    plt.subplot(111); plt.imshow(matches_posters_no_filtering[i])
    plt.tight_layout(); plt.show()
