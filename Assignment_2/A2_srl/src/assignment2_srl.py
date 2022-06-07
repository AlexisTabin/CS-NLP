# import sklearn_crfsuite
# from sklearn_crfsuite import metrics

from utils import load_data, print_statistics, preprocessing


def run_srl(args):
    # load training datasets and test datasets
    print("====> loading data (~= 1 minutes) <====")
    train_data, test_data = load_data(args.data_dir)

    # feature selection: print statistics for all features
    if args.print_stats:
        print("====> Training data <====")
        print_statistics(train_data)
        print("====> Test data <====")
        print_statistics(test_data)
        return

    # get features and labels
    ########################################################################
    #### This is the basic version: only three features are considered. ####
    ########################################################################
    print("====> proprocessing data (~= 1 minutes) <====")
    basic_features = ["LEMMA", "POS", "DEPREL"]
    train_features, train_labels = preprocessing(train_data, basic_features)
    test_features, test_labels = preprocessing(test_data, basic_features)

    # train the model:
    print("====> training your own model <====")
    ########################################################################
    ############# Please define and train your own model here. #############
    ########################################################################


    # test the model
    print("====> testing your own model <====")
    ########################################################################
    ######### Please test your trained model here: using F1-score. #########
    ########################################################################


    return