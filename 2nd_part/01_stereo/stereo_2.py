import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load both images
img_left = cv2.imread('left.png', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('right.png', cv2.IMREAD_GRAYSCALE)

# list of the points that correspond in the left and right images
pts = [
    ([841, 94], [813, 94]),
    ([740, 641], [685, 641]),
    ([732, 888], [705, 888]),
    ([1091, 805], [1065, 805]),
    ([781, 1068], [758, 1069]),
    ([1114, 669], [1075, 669]),
    ([1020, 771], [989, 771]),
    ([227, 922], [217, 922])
]

left_pts = np.asarray([x[0] for x in pts])
right_pts = np.asarray([x[1] for x in pts])

# compute the fundamental matrix
fundamental_mat, mask = cv2.findFundamentalMat(left_pts, right_pts, cv2.FM_LMEDS)

left_pts = left_pts[mask.ravel()==1]
right_pts = right_pts[mask.ravel()==1]

# find the epipolar lines (points in the left image lines are
# found in the corresponding line in the right image)
left_lines = cv2.computeCorrespondEpilines(left_pts.reshape(-1,1,2), 1, fundamental_mat)
left_lines = left_lines.reshape(-1,3)

def drawlines(img,lines,pts):
    ''' img - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img.shape
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    for r,pt in zip(lines,pts):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img = cv2.line(img, (x0,y0), (x1,y1), color,1)
        img = cv2.circle(img,tuple(pt),5,color,-1)
    return img

img_lines = drawlines(img_left, left_lines, left_pts)

plt.subplot(111),plt.imshow(img_lines)
plt.show()
