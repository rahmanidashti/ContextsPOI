import numpy as np
import scipy.sparse as sparse

from collections import defaultdict

from lib.AdaptiveKernelDensityEstimation import AdaptiveKernelDensityEstimation
from lib.SocialCorrelation import SocialCorrelation
from lib.CategoricalCorrelation import CategoricalCorrelation
from lib.AdditiveMarkovChain import AdditiveMarkovChain


def read_friend_data():
    social_data = open(social_file, 'r').readlines()
    social_matrix = np.zeros((user_num, user_num))
    for eachline in social_data:
        uid1, uid2 = eachline.strip().split()
        uid1, uid2 = int(uid1), int(uid2)
        social_matrix[uid1, uid2] = 1.0
        social_matrix[uid2, uid1] = 1.0
    return social_matrix


def read_poi_coos():
    poi_coos = {}
    poi_data = open(poi_file, 'r').readlines()
    for eachline in poi_data:
        lid, lat, lng = eachline.strip().split()
        lid, lat, lng = int(lid), float(lat), float(lng)
        poi_coos[lid] = (lat, lng)
    return poi_coos


def read_training_check_ins(training_matrix):
    check_in_data = open(check_in_file, 'r').readlines()
    training_check_ins = defaultdict(list)
    for eachline in check_in_data:
        uid, lid, ctime = eachline.strip().split()
        uid, lid, ctime = int(uid), int(lid), float(ctime)
        if not training_matrix[uid, lid] == 0:
            training_check_ins[uid].append([lid, ctime])
    return training_check_ins


def read_training_data():
    train_data = open(train_file, 'r').readlines()
    training_matrix = np.zeros((user_num, poi_num))
    sparse_training_matrix = sparse.dok_matrix((user_num, poi_num))
    for eachline in train_data:
        uid, lid, freq = eachline.strip().split()
        uid, lid, freq = int(uid), int(lid), int(freq)
        training_matrix[uid, lid] = freq
        sparse_training_matrix[uid, lid] = freq
    return training_matrix, sparse_training_matrix


def read_category_data():
    category_data = open(category_file, 'r').readlines()
    poi_cate_matrix = np.zeros((poi_num, category_num))
    for eachline in category_data:
        lid, cid = eachline.strip().split()
        lid, cid = int(lid), int(cid)
        poi_cate_matrix[lid, cid] = 1.0
    return poi_cate_matrix


def read_ground_truth():
    ground_truth = defaultdict(set)
    truth_data = open(test_file, 'r').readlines()
    for eachline in truth_data:
        uid, lid, _ = eachline.strip().split()
        uid, lid = int(uid), int(lid)
        ground_truth[uid].add(lid)
    return ground_truth


def main():
    training_matrix, sparse_training_matrix = read_training_data()
    training_check_ins = read_training_check_ins(sparse_training_matrix)
    sorted_training_check_ins = {uid: sorted(training_check_ins[uid], key=lambda k: k[1])
                                 for uid in training_check_ins}
    social_matrix = read_friend_data()
    ground_truth = read_ground_truth()

    poi_coos = read_poi_coos()
    poi_cate_matrix = read_category_data()

    AKDE.precompute_kernel_parameters(training_matrix, poi_coos)
    AMC.build_location_location_transition_graph(sorted_training_check_ins)
    SC.compute_beta(training_matrix, social_matrix)
    CC.compute_gamma(training_matrix, poi_cate_matrix)

    all_uids = list(range(user_num))
    all_lids = list(range(poi_num))
    np.random.shuffle(all_uids)

    Geo_results = np.zeros((user_num, poi_num))
    Temp_results = np.zeros((user_num, poi_num))
    So_results = np.zeros((user_num, poi_num))
    Cat_results = np.zeros((user_num, poi_num))

    for cnt, uid in enumerate(all_uids):
        if uid in ground_truth:
            for lid in all_lids:
                if training_matrix[uid, lid] == 0:
                    Geo_results[uid, lid] = AKDE.predict(uid, lid)
                    Temp_results[uid, lid] = AMC.predict(uid, lid)
                    So_results[uid, lid] = SC.predict(uid, lid)
                    Cat_results[uid, lid] = CC.predict(uid, lid)

    np.save("../model_combiner/contexts/" + dataset + "/G.npy", Geo_results)
    np.save("../model_combiner/contexts/" + dataset + "T.npy", Temp_results)
    np.save("../model_combiner/contexts/" + dataset + "S.npy", So_results)
    np.save("../model_combiner/contexts/" + dataset + "C.npy", Cat_results)


if __name__ == '__main__':
    data_dir = "../data/"
    dataset = "Yelp"

    size_file = data_dir + "Yelp_data_size.txt"
    check_in_file = data_dir + "Yelp_checkins.txt"
    train_file = data_dir + "Yelp_train.txt"
    tune_file = data_dir + "Yelp_tune.txt"
    test_file = data_dir + "Yelp_test.txt"
    social_file = data_dir + "Yelp_social_relations.txt"
    category_file = data_dir + "Yelp_poi_categories.txt"
    poi_file = data_dir + "Yelp_poi_coos.txt"

    user_num, poi_num, category_num = open(size_file, 'r').readlines()[0].strip('\n').split()
    user_num, poi_num, category_num = int(user_num), int(poi_num), int(category_num)

    top_k = 100

    AKDE = AdaptiveKernelDensityEstimation(alpha=0.5)
    AMC = AdditiveMarkovChain(delta_t=3600*24, alpha=0.05)
    SC = SocialCorrelation()
    CC = CategoricalCorrelation()

    main()
