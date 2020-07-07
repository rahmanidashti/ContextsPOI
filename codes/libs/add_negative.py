# add same negative number for train
# add N negative number for test

import random


class AddNegative:

    def __init__(self):
        """
        Initiate the positve and negative dictionary for users
        """
        self.pos_user_poi = dict()
        self.neg_user_poi = dict()

    def read_pos_samples(self, data_file):
        """
        This function read the positive instances (positive samples)
        to show the users visited locations.
        :param data_file: Positive instance sample file (eachline: uid lid freq)
        :return None: Nothing for return
        """
        sample_data = open(data_file, 'r').readlines()
        for visit in sample_data:
            uid, lid, freq = visit.split()
            uid, lid, freq = int(uid), int(lid), int(freq)
            if uid in self.pos_user_poi.keys():
                self.pos_user_poi[uid].add(lid)
            else:
                self.pos_user_poi[uid] = {lid}

    def create_negative_samples(self, neg_count, poi_num=0, is_train=True):
        """
        Here we create negative samples as for user based on their positive samples.
        As for training samples we add the same number of positive smaples to negative smaples
        :param neg_count: The number of negative samples that will be added for each users
        :param is_train: To differentaite between train and test (or other file)
        :return None: Nothing for return
        """
        for user, visited_pois in self.pos_user_poi.items():
            # Check if the file is a train file
            if is_train:
                neg_count = len(visited_pois)
            neg_counter = 0
            while neg_counter < neg_count:
                rand_poi = random.randint(0, (poi_num - 1))
                if rand_poi not in visited_pois:
                    if user in self.neg_user_poi.keys():
                        if rand_poi not in self.neg_user_poi[user]:
                            self.neg_user_poi[user].add(rand_poi)
                            neg_counter += 1
                    else:
                        self.neg_user_poi[user] = {rand_poi}
                        neg_counter += 1

    def write_sampls_files(self, dataset_name, is_train=True):
        """
        Here we write the new sample files with positive and negative instances
        :param dataset_name: The name of the dataset
        :param is_train: If true we write a train file otherwise the other file
        :return None: Nothing for return
        """
        if is_train:
            sample_data = open("pn_datasets/" + dataset_name + "_train_pn.txt", 'w')
        else:
            sample_data = open("pn_datasets/" + dataset_name + "_test_pn.txt", 'w')

        for user, visited_pois in self.pos_user_poi.items():
            for pos_poi in visited_pois:
                sample_data.write(str(user) + " " + str(pos_poi) + " " + str(1) + "\n")

        for user, unvisited_pois in self.neg_user_poi.items():
            for neg_poi in unvisited_pois:
                sample_data.write(str(user) + " " + str(neg_poi) + " " + str(0) + "\n")
