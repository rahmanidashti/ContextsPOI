import numpy as np
from decimal import *


def metric_evalator(metric_file, result_file):

    metric_data = open(metric_file, "r").readlines()

    precision, recall, nDCG, MAP = [], [], [], []

    for eachlie in metric_data:
        cnt, uid, precision_N, recall_N, nDCG_N, MAP_N = eachlie.strip().split()
        precision.append(Decimal(precision_N))
        recall.append(Decimal(recall_N))
        nDCG.append(Decimal(nDCG_N))
        MAP.append(Decimal(MAP_N))

    result_file.write('\t'.join(["precision", "recall", "nDCG", "MAP"]) + '\n')
    result_file.write('\t'.join([str(np.round(float(np.mean(precision)), decimals=4)),
                                 str(np.round(float(np.mean(recall)), decimals=4)),
                                 str(np.round(float(np.mean(nDCG)), decimals=4)),
                                 str(np.round(float(np.mean(MAP)), decimals=4))]) + '\n')


if __name__ == '__main__':

    # define the name of the directory to be created
    # datasets = ["Yelp", "Gowalla"]
    datasets = ["Yelp"]
    # models = ["N", "NC", "NG", "NS", "NT", "NGS", "NGT", "NGC", "NST", "NSC", "NTC", "NGST", "NGSC", "NSTC", "NGSTC"]
    models = ["NSTC"]
    top_n = [5, 10, 15, 20]

    for dataset in datasets:
        for model in models:
            if dataset == "Gowalla" and 'C' in model:
                continue
            else:
                for topn in top_n:
                    path = dataset + "/" + model + "/"
                    metric_file = path + model + "_top_" + str(topn) + ".txt"
                    result_file = open(path + model + "_mean_" + str(topn) + ".txt", 'w')
                    metric_evalator(metric_file=metric_file, result_file=result_file)