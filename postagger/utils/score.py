import numpy as np
from postagger.utils.params import Params
from copy import copy


def precision(predicted, true):
    """
    Returns the amount of relevant relevant retrieved documents divided by the amount of retrieved documents
    :param predicted: List of the predicted results
    :param true: List of the true labels
    :return: precision score
    """

    true_counter = 0
    retrieved_counter = 0

    if len(predicted) != len(true) or len(true) == 0 or len(predicted) == 0:
        return 0.0

    for i, value in enumerate(predicted):
        for word_idx, word_prediction in enumerate(value):
            if word_prediction == true[i][word_idx]:
                true_counter += 1
            retrieved_counter += 1

    return float(true_counter) / retrieved_counter


def recall(predicted, true):
    """
    Returns the amount of relevant relevant retrieved documents divided by the amount of relevant documents
    :param predicted: List of the predicted results
    :param true: List of the true labels
    :return: recall score
    """
    true_counter = 0
    relevant_counter = 0

    if len(predicted) != len(true) or len(true) == 0 or len(predicted) == 0:
        return 0.0

    for idx, prediction in enumerate(predicted):
        for word_id, word in enumerate(prediction):
            if word == true[idx][word_id]:
                true_counter += 1
            relevant_counter += 1

    return float(true_counter) / relevant_counter


def F1(predicted, true):
    """
    F1 calculated by the formula 2 * ((Precision * Recall) / (Precision + Recall)
    :param predicted: List of the predicted results
    :param true: List of the true labels
    :return: recall score
    """
    precision_score = precision(predicted, true)
    recall_score = recall(predicted, true)

    return 2 * ((precision_score * recall_score) / (precision_score + recall_score))


def accuracy(predicted, true):
    count_true = 0
    word_count = 0

    if len(predicted) != len(true) or len(true) == 0 or len(predicted) == 0:
        return 0.0

    for idx, item in enumerate(predicted):
        for word_id, word in enumerate(item):
            if word == true[idx][word_id]:
                count_true += 1
            word_count += 1

    return count_true / word_count


def confusion_matrix(predicted, true):
    pos_dic = copy(Params.pos_dic)
    len_pos = len(Params.poss)
    cm = np.zeros((len_pos, len_pos))

    for prediction_idx, prediction in enumerate(predicted):
        for word_id, word in enumerate(prediction):
            cm[pos_dic[true[prediction_idx][word_id]], pos_dic[word]] += 1

    return cm
