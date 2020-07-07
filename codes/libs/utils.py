from libs.add_negative import AddNegative
from libs.evaluation import Evaluation
import time
import numpy as np


class Utils:

    def load_embeddings(self, path):
        ctime = time.time()
        print("Loading U and L...", )
        user_embeddings = np.load(path + "U.npy")
        poi_embeddings = np.load(path + "L.npy")
        print("Done. Elapsed time:", time.time() - ctime, "s")

        return user_embeddings, poi_embeddings

    def dataset_info(self, dataset_name):
        dataset_dir = "../" + dataset_name + "/"

        size_file = dataset_dir + dataset_name + "_data_size.txt"
        train_file = dataset_dir + dataset_name + "_train.txt"
        test_file = dataset_dir + dataset_name + "_test.txt"

        user_num, poi_num, _ = open(size_file, 'r').readlines()[0].strip('\n').split()
        user_num, poi_num = int(user_num), int(poi_num)

        return size_file, train_file, test_file, user_num, poi_num

    def add_negative_run(self, dataset_name: str, data_file: str, neg_count: int, poi_num: int, is_train: bool) -> None:
        negative_adder = AddNegative()
        negative_adder.read_pos_samples(data_file)
        negative_adder.create_negative_samples(neg_count=neg_count, poi_num=poi_num, is_train=is_train)
        negative_adder.write_sampls_files(dataset_name=dataset_name, is_train=is_train)

        print("The data file with negative instances has been writen.")

    def write_results(self, data_file: str, prob_predictions: np.ndarray) -> None:

        out_prob = open("results/UserPOI_Prob.txt", 'w')

        for index, test_instance in enumerate(data_file):
            uid, lid, _ = test_instance.strip().split()
            uid, lid = int(uid), int(lid)

            out_prob.write(str(uid) + " " + str(lid) + " " + str(prob_predictions[index]) + "\n")

    def evaluation(self, train_file, test_file, prob_file, user_num, poi_num):
            eval = Evaluation()
            eval.eval(train_file, test_file, prob_file, user_num, poi_num)

