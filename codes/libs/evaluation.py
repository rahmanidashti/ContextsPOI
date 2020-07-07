import numpy as np
import scipy.sparse as sparse

from collections import defaultdict

from libs.metrics import precisionk, recallk, ndcgk, mapk


class Evaluation:

    def read_training_data(self, train_file, user_num, poi_num):
        train_data = open(train_file, 'r').readlines()
        sparse_training_matrix = sparse.dok_matrix((user_num, poi_num))
        training_tuples = set()
        for eachline in train_data:
            uid, lid, freq = eachline.strip().split()
            uid, lid, freq = int(uid), int(lid), int(freq)
            sparse_training_matrix[uid, lid] = freq
            training_tuples.add((uid, lid))
        return sparse_training_matrix, training_tuples

    def read_ground_truth(self, test_file):
        ground_truth = defaultdict(set)
        truth_data = open(test_file, 'r').readlines()
        for eachline in truth_data:
            uid, lid, _ = eachline.strip().split()
            uid, lid = int(uid), int(lid)
            ground_truth[uid].add(lid)
        return ground_truth

    def read_results_prob(self, prob_file):
        prob_results = dict()
        results_data = open(prob_file, 'r').readlines()
        for eachline in results_data:
            uid, lid, prob = eachline.strip().split()
            prob = prob.replace("[", "").replace("]", "")
            uid, lid, prob = int(uid), int(lid), float(prob)
            prob_results[(uid, lid)] = prob

        return prob_results

    def eval(self, train_file, test_file, prob_file, user_num, poi_num):
        sparse_training_matrix, training_tuples = self.read_training_data(train_file, user_num, poi_num)
        ground_truth = self.read_ground_truth(test_file=test_file)
        prob_results = self.read_results_prob(prob_file=prob_file)

        rec_list = open("results/reclist_top_" + str(100) + ".txt", 'w')
        result_5 = open("results/result_top_" + str(5) + ".txt", 'w')
        result_10 = open("results/result_top_" + str(10) + ".txt", 'w')
        result_15 = open("results/result_top_" + str(15) + ".txt", 'w')
        result_20 = open("results/result_top_" + str(20) + ".txt", 'w')

        all_uids = list(range(user_num))
        all_lids = list(range(poi_num))
        np.random.shuffle(all_uids)

        # list for different ks
        precision_5, recall_5, nDCG_5, MAP_5 = [], [], [], []
        precision_10, recall_10, nDCG_10, MAP_10 = [], [], [], []
        precision_15, recall_15, nDCG_15, MAP_15 = [], [], [], []
        precision_20, recall_20, nDCG_20, MAP_20 = [], [], [], []

        for cnt, uid in enumerate(all_uids):
            if uid in ground_truth:
                overall_scores = []
                for lid in all_lids:
                    if (uid, lid) not in training_tuples:
                        if (uid, lid) in prob_results.keys():
                            overall_scores.append(prob_results[(uid, lid)])
                        else:
                            overall_scores.append(0)
                    else:
                        overall_scores.append(-1)

                overall_scores = np.array(overall_scores)

                predicted = list(reversed(overall_scores.argsort()))[:100]
                actual = ground_truth[uid]

                # calculate the average of different k
                precision_5.append(precisionk(actual, predicted[:5]))
                recall_5.append(recallk(actual, predicted[:5]))
                nDCG_5.append(ndcgk(actual, predicted[:5]))
                MAP_5.append(mapk(actual, predicted[:5], 5))

                precision_10.append(precisionk(actual, predicted[:10]))
                recall_10.append(recallk(actual, predicted[:10]))
                nDCG_10.append(ndcgk(actual, predicted[:10]))
                MAP_10.append(mapk(actual, predicted[:10], 10))

                precision_15.append(precisionk(actual, predicted[:15]))
                recall_15.append(recallk(actual, predicted[:15]))
                nDCG_15.append(ndcgk(actual, predicted[:15]))
                MAP_15.append(mapk(actual, predicted[:15], 15))

                precision_20.append(precisionk(actual, predicted[:20]))
                recall_20.append(recallk(actual, predicted[:20]))
                nDCG_20.append(ndcgk(actual, predicted[:20]))
                MAP_20.append(mapk(actual, predicted[:20], 20))

                print(cnt, uid, "pre@10:", np.mean(precision_10), "rec@10:", np.mean(recall_10))

                rec_list.write('\t'.join([
                    str(cnt),
                    str(uid),
                    ','.join([str(lid) for lid in predicted])
                ]) + '\n')

                # write the different ks
                result_5.write('\t'.join([str(cnt), str(uid), str(np.mean(precision_5)), str(np.mean(recall_5)),
                                          str(np.mean(nDCG_5)), str(np.mean(MAP_5))]) + '\n')
                result_10.write('\t'.join([str(cnt), str(uid), str(np.mean(precision_10)), str(np.mean(recall_10)),
                                           str(np.mean(nDCG_10)), str(np.mean(MAP_10))]) + '\n')
                result_15.write('\t'.join([str(cnt), str(uid), str(np.mean(precision_15)), str(np.mean(recall_15)),
                                           str(np.mean(nDCG_15)), str(np.mean(MAP_15))]) + '\n')
                result_20.write('\t'.join([str(cnt), str(uid), str(np.mean(precision_20)), str(np.mean(recall_20)),
                                           str(np.mean(nDCG_20)), str(np.mean(MAP_20))]) + '\n')

        print("<< Task Finished >>")
