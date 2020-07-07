from libs import nn_models
from libs.utils import Utils


if __name__ == '__main__':

    # Dataset names
    dataset_name = 'Yelp'

    # Embedding paths
    embeddings_path = "embeddings/" + dataset_name + "/"

    # Class instances
    utils: Utils = Utils()
    nn_models = nn_models.NeuralNetwork()

    # Load users and POIs embeddings
    user_embeddings, poi_embeddings = utils.load_embeddings(path=embeddings_path)

    # Load datasets and its info
    size_file, train_file, test_file, user_num, poi_num = utils.dataset_info(dataset_name=dataset_name)

    # Create train data with negative samples
    utils.add_negative_run(
        dataset_name=dataset_name,
        data_file=train_file,
        neg_count=0,
        poi_num=poi_num,
        is_train=True)

    # Create test data with negative samples
    utils.add_negative_run(
        dataset_name=dataset_name,
        data_file=test_file,
        neg_count=100,
        poi_num=poi_num,
        is_train=False)

    pn_train_file = "pn_datasets/" + dataset_name + "_train_pn.txt"
    pn_test_file = "pn_datasets/" + dataset_name + "_test_pn.txt"

    train_user_input, train_poi_input, train_labels, train_data = nn_models.read_nn_data(
        data_file=pn_train_file,
        user_embeddings=user_embeddings,
        poi_embeddings=poi_embeddings)

    model = nn_models.NCF()

    nn_models.run_nn_model(model=model,
                           user_input=train_user_input,
                           poi_input=train_poi_input,
                           labels=train_labels, epochs=1)

    test_user_input, test_poi_input, test_labels, test_data = nn_models.read_nn_data(
        data_file=pn_test_file,
        user_embeddings=user_embeddings,
        poi_embeddings=poi_embeddings)

    model.evaluate([test_user_input, test_poi_input], test_labels)
    prob_predictions = model.predict([test_user_input, test_poi_input])

    utils.write_results(data_file=test_data, prob_predictions=prob_predictions)

    prob_file = "results/UserPOI_Prob.txt"

    utils.evaluation(train_file=train_file,
                     test_file=test_file,
                     prob_file=prob_file,
                     user_num=user_num,
                     poi_num=poi_num)
