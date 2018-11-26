from scipy.sparse import csr_matrix
import numpy as np


def read_file(file_path):
    """
    Securely read a file from a given path
    :param file_path: Path of the file
    :return: The content of the text file
    """
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print('Could not find file in the path')
    except Exception as e:
        print(e)


def create_all_features(feature_functions, feature_lst_functions, sentence_tuples, sentence):
    """
    This function create all of the features implemented in the features class
    :param feature_lst_functions: List of functions that returns lists
    :param feature_functions:  List of all of the feature functions
    :param sentence_tuples: Tuples with all of the threes of labels
    :param sentence: The text of the sentence
    :return: Sparse matrix with the X
    """

    # x_sentence = csr_matrix()
    x_sentence = []

    for feature_function in feature_functions:
        x_sentence.append(feature_function(tuples=sentence_tuples, text=sentence))
        # x_tmp = np.array(feature_function(tuples=sentence_tuples, text=sentence))
        # print(x_tmp.shape)

    for feature_lst_function in feature_lst_functions:
        x_sentence += feature_lst_function(tuples=sentence_tuples, text=sentence)

    return np.array(x_sentence).T
