import scipy.sparse as sparse
from lib.PoissonFactorModel import PoissonFactorModel


def read_training_data():
    train_data = open(train_file, 'r').readlines()
    sparse_training_matrix = sparse.dok_matrix((user_num, poi_num))
    training_tuples = set()
    for eachline in train_data:
        uid, lid, freq = eachline.strip().split()
        uid, lid, freq = int(uid), int(lid), int(freq)
        sparse_training_matrix[uid, lid] = freq
        training_tuples.add((uid, lid))
    return sparse_training_matrix, training_tuples


def run():
    sparse_training_matrix, training_tuples = read_training_data()

    PFM.train(sparse_training_matrix, max_iters=50, learning_rate=1e-4)
    PFM.save_model(path_to_save)


if __name__ == '__main__':

    datasets = ['Gowalla', 'Yelp']
    user_num, poi_num = None, None

    for dataset in datasets:
        # path to save the embeddings
        path_to_save = "../embeddings/" + dataset + "/"
        # dir to datasets
        data_dir = "../../datasets/" + dataset + "/"
        size_file = data_dir + dataset + "_data_size.txt"
        train_file = data_dir + dataset + "_train.txt"

        if dataset == "Gowalla":
            user_num, poi_num = open(size_file, 'r').readlines()[0].strip('\n').split()
        elif dataset == "Yelp":
            user_num, poi_num, _ = open(size_file, 'r').readlines()[0].strip('\n').split()

        user_num, poi_num = int(user_num), int(poi_num)

        PFM = PoissonFactorModel(K=30, alpha=20.0, beta=0.2)

        run()
