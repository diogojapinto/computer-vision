import numpy as np
from sklearn.metrics import confusion_matrix
import cv2
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.base import clone

# load features from csv
samples = np.genfromtxt('spambase.txt', delimiter=';')

features = samples[:, :-1]
label = samples[:, -1]

train = range(3000)
test = range(3000, len(features))

y_true = label[test]

def get_conf_mat(raw_clf):
    clf = clone(raw_clf)
    clf.fit(features[train], label[train])
    y_pred = clf.predict(features[test])

    conf_mat = confusion_matrix(y_true, y_pred)
    return conf_mat

########################
# Gaussian Naive Bayes #
########################

bayes_clf = GaussianNB()
conf_mat_bayes = get_conf_mat(bayes_clf)

##########################
# Support Vector Machine #
##########################

svc_clf = SVC()
conf_mat_svc = get_conf_mat(svc_clf)

print('Naive Bayes')
print(conf_mat_bayes)
print()
print('SVM')
print(conf_mat_svc)
