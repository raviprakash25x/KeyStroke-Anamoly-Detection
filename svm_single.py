import pandas as pd
from sklearn.cluster import KMeans
import math
from sklearn.svm import OneClassSVM
import numpy as np
from sklearn.metrics import roc_curve

msr = []
far = []

def getEER(user_scores, imposter_scores):
    #print user_scores
    labels = [0]*len(user_scores) + [1]*len(imposter_scores)
    fpr, tpr, thresholds = roc_curve(labels, user_scores + imposter_scores)
    missrates = 1 - tpr
    farates = fpr
    msr.append(np.mean(missrates))
    far.append(np.mean(farates))
    dists = missrates - farates
    idx1 = np.argmin(dists[dists >= 0])
    idx2 = np.argmax(dists[dists < 0])
    x = [missrates[idx1], farates[idx1]]
    y = [missrates[idx2], farates[idx2]]
    a = ( x[0] - x[1] ) / ( y[1] - x[1] - y[0] + x[0] )
    eer = x[0] + a * ( y[0] - x[0] )
    return eer

data = pd.read_csv("/home/ravi/Documents/DWDM/Project/data/DSL-StrongPasswordData.csv")
subjects = data["subject"].unique()
EERS = []

for subject in subjects:
    genuine = data.loc[data.subject == subject, "H.period":"H.Return"]
    genuine_train = genuine.head(300)
    genuine_test = genuine.tail(100)
    imposter = data.loc[data.subject != subject, "H.period":"H.Return"]
    test = genuine_test.append(imposter)
    clf = OneClassSVM(kernel='rbf', gamma=26)
    clf.fit(genuine_train)
    user_scores = -clf.decision_function(genuine_test)
    imposter_scores = -clf.decision_function(imposter)
    #print user_scores
    #print imposter_scores
    curr_eer = getEER(list(user_scores), list(imposter_scores))
    EERS.append(curr_eer)
    print "Subject:", subject, "EER: ", curr_eer
    #exit(0)

print "\n\nOverall Equal Error Rate:",np.mean(EERS)
print "Miss Rate:", np.mean(msr)
print "False Rate: ",np.mean(far)
