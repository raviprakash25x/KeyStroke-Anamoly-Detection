import pandas as pd
from sklearn.cluster import KMeans
import math


data = pd.read_csv("/home/ravi/Documents/DWDM/Project/data/DSL-StrongPasswordData.csv")
subjects = data["subject"].unique()
min_distance = 0.28
tot_correct = 0
tot_incorrect = 0
misses = 0
false_alarms = 0
total_imposter = 0
total_genuine = 0

def getDistance(a, b):
    ret = reduce(lambda x,y: x + pow((a[y]-b[y]), 2),range(len(a)),0.0)
    return math.sqrt(ret)

for subject in subjects:
    # taking only current subject as genuine
    #ignoring first two columns as useless
    genuine = data.loc[data.subject == subject, "H.period":"H.Return"]
    genuine_train = genuine.head(300)
    genuine_test = genuine.tail(100)
    #taking all others as imposter
    imposter = data.loc[data.subject != subject, "H.period":"H.Return"]
    imposter = imposter[:200]
    cluster = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                                     precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)
    cluster.fit(genuine_train)
    #result = cluster.predict(imposter)
    #print cluster.cluster_centers_
    #print result
    centroids = cluster.cluster_centers_
    imposter = imposter.values
    genuine_test = genuine_test.values

    #print len(genuine_train)
    #print len(imposter)
    #print len(genuine_test)
    correct = 0
    incorrect = 0
    total_imposter += len(imposter)
    total_genuine += len(genuine_test)

    for i in range(len(imposter)):
        count  =0

        for j in range(len(centroids)):
            dis = getDistance(imposter[i], centroids[j])

            if dis < min_distance:
                count += 1

        if count > 0:
            incorrect += 1
            misses += 1
        else:
            correct += 1

    for i in range(len(genuine_test)):
        count  =0

        for j in range(len(centroids)):
            dis = getDistance(genuine_test[i], centroids[j])

            if dis < min_distance:
                count += 1

        if count > 0:
            correct += 1
        else:
            incorrect += 1
            false_alarms += 1

    print  "Subject:", subject, "Correct: " ,correct, "Incorrect: ", incorrect
    tot_correct += correct
    tot_incorrect += incorrect
    #exit(0)

print "\n\nTotal Correct", tot_correct
print "Total Incorrect", tot_incorrect
print "Accuracy:", (tot_correct*100.0)/(tot_correct+tot_incorrect)
print "Misses:", misses
print "False Alarms:",false_alarms