import numpy as np
import os
from collections import defaultdict
from itertools import chain, combinations
from libs.metrics import precisionk, recallk, ndcgk, mapk


def find_subsets(iterable):
    """
    If you don't like that empty tuple at the beginning, you can just change the range
    statement to range(1, len(s)+1) to avoid a 0-length combination.
    :param iterable:
    :return:
    """
    " powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def read_training_data():
    # load train data
    train_data = open(train_file, 'r').readlines()
    # sparse_training_matrix = sparse.dok_matrix((user_num, poi_num))
    training_tuples = set()
    for eachline in train_data:
        uid, lid, freq = eachline.strip().split()
        uid, lid, freq = int(uid), int(lid), int(freq)
        # sparse_training_matrix[uid, lid] = freq
        training_tuples.add((uid, lid))

    return training_tuples


def read_ground_truth():
    ground_truth = defaultdict(set)
    truth_data = open(test_file, 'r').readlines()
    for eachline in truth_data:
        uid, lid, _ = eachline.strip().split()
        uid, lid = int(uid), int(lid)
        ground_truth[uid].add(lid)
    print("The loading of Ground Truth Finished.")
    return ground_truth


def main():
    training_tuples = read_training_data()
    ground_truth = read_ground_truth()

    result_5 = open(folder_path + "/" + model_name + "_top_" + str(5) + ".txt", 'w')
    result_10 = open(folder_path + "/" + model_name + "_top_" + str(10) + ".txt", 'w')
    result_15 = open(folder_path + "/" + model_name + "_top_" + str(15) + ".txt", 'w')
    result_20 = open(folder_path + "/" + model_name + "_top_" + str(20) + ".txt", 'w')

    all_uids = list(range(user_num))
    all_lids = list(range(poi_num))
    np.random.shuffle(all_uids)

    # list for different ks
    precision_5, recall_5, nDCG_5, MAP_5 = 0, 0, 0, 0
    precision_10, recall_10, nDCG_10, MAP_10 = 0, 0, 0, 0
    precision_15, recall_15, nDCG_15, MAP_15 = 0, 0, 0, 0
    precision_20, recall_20, nDCG_20, MAP_20 = 0, 0, 0, 0

    for cnt, uid in enumerate(all_uids):
        print("Evaluate on sample " + str(cnt))
        if uid in ground_truth:
            # What is the meaning of the following structure?
            overall_scores = [
                Neu_results[uid, lid] + So_results[uid, lid] + Temp_results[uid, lid] + Cat_results[uid, lid]
                if (uid, lid) not in training_tuples else -1
                for lid in all_lids]

            overall_scores = np.array(overall_scores)

            predicted = list(reversed(overall_scores.argsort()))[:top_k]
            actual = ground_truth[uid]

            # calculate the average of different k
            precision_5 = precisionk(actual, predicted[:5])
            recall_5 = recallk(actual, predicted[:5])
            nDCG_5 = ndcgk(actual, predicted[:5])
            MAP_5 = mapk(actual, predicted[:5], 5)

            precision_10 = precisionk(actual, predicted[:10])
            recall_10 = recallk(actual, predicted[:10])
            nDCG_10 = ndcgk(actual, predicted[:10])
            MAP_10 = mapk(actual, predicted[:10], 10)

            precision_15 = precisionk(actual, predicted[:15])
            recall_15 = recallk(actual, predicted[:15])
            nDCG_15 = ndcgk(actual, predicted[:15])
            MAP_15 = mapk(actual, predicted[:15], 15)

            precision_20 = precisionk(actual, predicted[:20])
            recall_20 = recallk(actual, predicted[:20])
            nDCG_20 = ndcgk(actual, predicted[:20])
            MAP_20 = mapk(actual, predicted[:20], 20)

            # write the different ks
            result_5.write(
                '\t'.join([str(cnt), str(uid), str(precision_5), str(recall_5), str(nDCG_5), str(MAP_5)]) + '\n')
            result_10.write(
                '\t'.join([str(cnt), str(uid), str(precision_10), str(recall_10), str(nDCG_10), str(MAP_10)]) + '\n')
            result_15.write(
                '\t'.join([str(cnt), str(uid), str(precision_15), str(recall_15), str(nDCG_15), str(MAP_15)]) + '\n')
            result_20.write(
                '\t'.join([str(cnt), str(uid), str(precision_20), str(recall_20), str(nDCG_20), str(MAP_20)]) + '\n')

    print("<< Finished >>")


if __name__ == '__main__':

    # define the name of the directory to be created
    dataset_name = "Yelp"
    model_name = "NSTC"
    folder_path = dataset_name + "/" + model_name

    if dataset_name == "Yelp":
        user_num = 7135
        poi_num = 16621
    else:
        user_num = 5628
        poi_num = 31803

    top_k = 100

    try:
        os.mkdir(folder_path)
    except OSError:
        print("Creation of the directory %s failed" % folder_path)
    else:
        print("Successfully created the directory %s " % folder_path)

    train_file = "../../datasets/" + dataset_name + "/" + dataset_name + "_train.txt"
    test_file = "../../datasets/" + dataset_name + "/" + dataset_name + "_test.txt"

    print("Start to read the files ...")
    # adding the contextual information
    Neu_results = np.load("contexts/" + dataset_name + "/" + "N.npy")
    Geo_results = np.load("contexts/" + dataset_name + "/" + "G.npy")
    So_results = np.load("contexts/" + dataset_name + "/" + "S.npy")
    Temp_results = np.load("contexts/" + dataset_name + "/" + "T.npy")
    if dataset_name == "Yelp":
        Cat_results = np.load("contexts/" + dataset_name + "/" + "C.npy")

    main()