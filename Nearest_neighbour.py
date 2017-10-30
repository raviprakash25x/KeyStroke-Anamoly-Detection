import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve

data = pd.read_csv("/home/ravi/Documents/DWDM/Project/data/DSL-StrongPasswordData.csv")
subjects = data["subject"].unique()
EERS = []

def evaluateEER(user_scores, imposter_scores):
    labels = [0]*len(user_scores) + [1]*len(imposter_scores)
    fpr, tpr, thresholds = roc_curve(labels, user_scores + imposter_scores)
    missrates = 1 - tpr
    farates = fpr
    dists = missrates - farates
    idx1 = np.argmin(dists[dists >= 0])
    idx2 = np.argmax(dists[dists < 0])
    x = [missrates[idx1], farates[idx1]]
    y = [missrates[idx2], farates[idx2]]
    a = ( x[0] - x[1] ) / ( y[1] - x[1] - y[0] + x[0] )
    eer = x[0] + a * ( y[0] - x[0] )
    return eer

def test(covariance, genuine_train, test):
    scores = []

    for i in range(test.shape[0]): #no. of rows
        curr = []

        for j in range(genuine_train.shape[0]):
            difference = test.iloc[i] - genuine_train.iloc[j]
            x = np.dot(np.dot(difference.T, covariance), difference)
            curr.append(x)
        scores.append(min(curr))

    return scores

for subject in subjects:
    # taking only current subject as genuine
    # ignoring first two columns as useless
    genuine = data.loc[data.subject == subject, "H.period":"H.Return"]
    genuine_train = genuine.head(300)
    genuine_test = genuine.tail(100)
    # taking all others as imposter
    imposter = data.loc[data.subject != subject, "H.period":"H.Return"]
    #imposter = imposter[:200]
    covariance = np.linalg.inv(np.cov(genuine_train.T))
    #user_score, imposter_score  = test(covariance, genuine_train, genuine_test, imposter)
    genuineUser_scores = test(covariance, genuine_train, genuine_test)
    imposter_scores = test(covariance, genuine_train, imposter)
    curr_eer = evaluateEER(genuineUser_scores, imposter_scores)
    EERS.append(curr_eer)
    print subject, ":", curr_eer

print np.mean(EERS)