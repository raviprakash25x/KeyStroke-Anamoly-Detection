import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve

data = pd.read_csv("/home/ravi/Documents/DWDM/Project/data/DSL-StrongPasswordData.csv")
subjects = data["subject"].unique()
EERS = []


def getEER(user_scores, imposter_scores):
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

def test(mean, std_dev, test):
    scores = []

    for i in range(test.shape[0]):
        count = 0.0

        for j in range(len(mean)):
            curr = abs(test.iloc[i].values[j] - mean[j]) // std_dev[j]

            if(curr > 2.96):
                count += 1.0

            scores.append(count)

    return scores

for subject in subjects:
    genuine = data.loc[data.subject == subject, "H.period":"H.Return"]
    genuine_train = genuine.head(300)
    genuine_test = genuine.tail(100)
    # taking all others as imposter
    imposter = data.loc[data.subject != subject, "H.period":"H.Return"]
    mean_train = genuine_train.mean().values
    std_dev_train = genuine_train.std().values
    genuine_user_scores = test(mean_train, std_dev_train, genuine_test)
    imposter_scores = test(mean_train, std_dev_train, imposter)
    curr_eer = getEER(genuine_user_scores, imposter_scores)
    print subject, ":", curr_eer
    EERS.append(curr_eer)

print EERS.mean()