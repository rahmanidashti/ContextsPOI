import tensorflow as tf
import numpy as np
import random


class NeuralNetwork:

    def read_nn_data(self, data_file, user_embeddings, poi_embeddings):
        sample_data = open(data_file, 'r').readlines()
        random.shuffle(sample_data)

        # input data structure
        labels = []
        user_input = []
        poi_input = []

        for sample_instance in sample_data:
            uid, lid, visit = sample_instance.strip().split()
            uid, lid, visit = int(uid), int(lid), int(visit)

            # add to user_input list
            user_input.append(user_embeddings[uid])

            # add to poi_input list
            poi_input.append(poi_embeddings[lid])

            # set to labels
            labels.append(visit)

        # conver list to numpy array
        user_input = np.array(user_input)
        poi_input = np.array(poi_input)
        labels = np.array(labels)
        labels = labels.reshape(-1, 1)

        return user_input, poi_input, labels, sample_data

    def NCF(self):
        user_embedding = tf.keras.Input(shape=(30,), name='user_embedding')
        poi_embedding = tf.keras.Input(shape=(30,), name='poi_embedding')

        user_poi_embedding = tf.keras.layers.concatenate([user_embedding, poi_embedding])

        hidden_layer_1 = tf.keras.layers.Dense(128, activation=tf.nn.sigmoid)(user_poi_embedding)
        droupout_layer_1 = tf.keras.layers.Dropout(.2)(hidden_layer_1)
        hidden_layer_2 = tf.keras.layers.Dense(64, activation=tf.nn.sigmoid)(droupout_layer_1)

        output_layer = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(hidden_layer_2)

        model = tf.keras.Model(inputs=[user_embedding, poi_embedding], outputs=[output_layer])

        return model

    def ConvNCF(self) -> tf.keras.Model:
        user_embedding = tf.keras.Input(shape=(30, 1), name='user_embedding')
        poi_embedding = tf.keras.Input(shape=(30, 1), name='poi_embedding')

        # Outer product
        relation = tf.linalg.matmul(tf.transpose(user_embedding, perm=[0, 2, 1]),
                                    tf.transpose(poi_embedding, perm=[0, 2, 1]),
                                    transpose_a=True)
        # print(relation.shape)
        net_input = tf.expand_dims(relation, -1)

        # CNN
        conv_layer = tf.keras.layers.Conv2D(64, (5, 5), activation=tf.nn.sigmoid)(net_input)
        maxpool_layer = tf.keras.layers.MaxPooling2D(2, 2)(conv_layer)
        flat_layer = tf.keras.layers.Flatten()(maxpool_layer)

        hidden_layer_1 = tf.keras.layers.Dense(128, activation=tf.nn.sigmoid)(flat_layer)
        # droupout_layer_1 = tf.keras.layers.Dropout(.2)(hidden_layer_1)
        hidden_layer_2 = tf.keras.layers.Dense(64, activation=tf.nn.sigmoid)(hidden_layer_1)

        output_layer = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(hidden_layer_2)

        model = tf.keras.Model(inputs=[user_embedding, poi_embedding], outputs=[output_layer])

        return model

    def run_nn_model(self, model, user_input, poi_input, labels, epochs):
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])

        model.fit([user_input, poi_input], labels, epochs=epochs)
