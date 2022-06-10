import sklearn_crfsuite

from utils import load_data, print_statistics, preprocessing
import itertools
import threading
import time
import sys
from sklearn_crfsuite.metrics import flat_f1_score, flat_precision_score, flat_recall_score, flat_classification_report


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
            
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )

    crf.fit(train_features, train_labels)


    print("====> testing your own model <====")
    # # test the model
    # ########################################################################
    # ######### Please test your trained model here: using F1-score. #########
    # ########################################################################

    y_pred = crf.predict(test_features)
    print("y_pred: ", y_pred)

    f1_score = flat_f1_score(test_labels,y_pred,
                        average='weighted')
    print("f1_score: ", f1_score)

    precision = flat_precision_score(test_labels,y_pred)
    print("precision: ", precision)

    recall = flat_recall_score(test_labels,y_pred)
    print("recall: ", recall)

    report = flat_classification_report(test_labels,y_pred)
    print("report: ", report)

    return