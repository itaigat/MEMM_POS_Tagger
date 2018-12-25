import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from postagger.utils.common import timeit
from time import time
import operator
from postagger.utils.features import build_y_x_matrix, build_feature_matrix_, init_callable_features
from postagger.utils.params import Params
from postagger.utils.common import viterbi_s, get_poss_dict
from postagger.utils.common import pickle_save, pickle_load
from postagger.utils.score import accuracy, get_top_k_errors
from postagger.utils.score import tag_test_file


class MaximumEntropyClassifier:
    """
    implements MEMM, training and inference methods
    """

    @timeit
    def __init__(self, iterable_sentences, preprocess_dict, poss):
        """
        iterable sentences:
            a tuple (tuples, tags, stripped_sentence)

            tuples: [('*', '*', 0, 0), ('*', 'DT', 0, 1),..]
            tags: ['DT', 'NNP',...]
            stripped_sentence: ['All', 'Nasdaq', ..]

            tuples are history tuples <u,v,sentence,i> ,
            where sentence is an id, refers to its position on the corpus, e.g., first sentence is 0.
        """
        print("Init classifier")
        # prepare for converting x,y (history-tuple and tag)
        # into features matrix
        X, y, sentences = [], [], []
        for tuples, tags, sentence in iterable_sentences:
            for i in range(len(tuples)):
                X.append(tuples[i])
                y.append(tags[i])
            sentences.append(sentence)

        self.callable_functions = init_callable_features(poss, Params, preprocess_dict)

        # build matrices
        t1 = time()
        self.feature_matrix = build_feature_matrix_(X, y, sentences, self.callable_functions)
        print("Building feature matrix: %f s" % (time() - t1))

        t2 = time()
        self.y_x_matrix, self.help_matrix = build_y_x_matrix(X, poss, sentences, self.callable_functions)
        print("Building y_x features matrix: %f s" % (time() - t2))

        self.X = X
        self.y = y
        self.sentences = sentences
        self.reg = 0
        self.v = None
        self.verbose = 0
        self.poss = poss

    @timeit
    def fit(self, reg=0, verbose=0):
        self.verbose = verbose
        self.reg = reg
        m = self.feature_matrix.shape[1]
        v_init = np.zeros(m)

        res = fmin_l_bfgs_b(func=self.loss, x0=v_init, fprime=self.gradient)

        # save normalized parameters vector
        # (normalization is needed as numeric fix for viterbi computations)
        v = res[0]
        self.v = v / v.sum()

        print("W dim: " + str(v.shape))
        print("W zeros: %d" % np.sum(v == 0), "W pos: %d" % np.sum(v > 0), "W neg: %d" % np.sum(v < 0))

    def loss(self, v):
        a = self.feature_matrix.dot(v).sum()
        b = np.sum(np.log(self.help_matrix * np.exp(self.y_x_matrix * v)))
        c = 0.5 * self.reg * np.dot(v.transpose(), v)

        res = - (a - b - c)

        print("Loss: %f" % res)

        return res

    def gradient(self, v):
        sums = 1.0 / (self.help_matrix.transpose() * (self.help_matrix * np.exp(self.y_x_matrix * v)))
        emp_counts = self.feature_matrix.sum(axis=0)
        exp_counts = (self.y_x_matrix.transpose() * (sums * np.exp(self.y_x_matrix * v))).T
        reg_grad = self.reg * v.T
        res = -(emp_counts - exp_counts - reg_grad)

        print("Grad: %f" % (res*res.T)[0])

        return res

    def predict(self, X, comp=False):
        """
        :param X: [(sentence_index, sentence_list),...,()]
        :return: [tags_sentence_1...]
        """
        tags_predicted = []
        sentences = []
        true_tags = []

        for tuples, tags, sentence in X:
            sentences.append(sentence)

        # Params.q = calc_q(X)

        if self.v is None:
            print('You need to fit the model before prediction')
            return None
        else:
            for tuples, tags, sentence in X:
                tmp = viterbi_s(tuples[0][2], sentences, self.v, self.callable_functions, self.poss)
                tags_predicted.append(tmp)
                true_tags.append(tags)
                if not comp:
                    acc = accuracy(tags_predicted, true_tags)
                    print("Sentence: " + str(sentence))
                    print("Tags: " + str(tags))
                    print("Pred: " + str(tmp))
                    print('Accuracy: ', acc)
        if not comp:
            print('Total accuracy: ', acc)
            top_k_errors = get_top_k_errors(tags_predicted, true_tags, get_poss_dict(self.poss), k=10)
            print('Top K errors: ', top_k_errors)
        if not comp:
            return {'accuracy': acc, 'prediction': tags_predicted, 'top_10_errors': top_k_errors}
        else:
            return {'prediction': tags_predicted}

    def get_num_features(self):
        return self.feature_matrix.shape[1]

    def get_enabled_features_per_tag(self):
        y_x_sum_rows = self.y_x_matrix.sum(axis=1)
        enabled = y_x_sum_rows.reshape(len(self.X), len(self.poss)).sum(axis=0)
        counts = enabled.tolist()[0]
        tuples_list = []
        for i, count in enumerate(counts):
            tup = (self.poss[i], count)
            tuples_list.append(tup)

        counts_dict = dict(tuples_list)

        sorted_dict = sorted(counts_dict.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_dict


def save_load_init_model(clf, filename):
    """saves or loads clf model
    clf: classifier object after running constructor
    filename: if empty then function assumes user wants to load"""
    # try loading
    if clf is None:
        clf = pickle_load(filename)
        if clf is not None:
            print("Loaded classifier object")
            return clf
        else:
            print("Classifier object loading failed")
    # save
    else:
        ret = pickle_save(clf, filename)
        if ret:
            print("Classifier object saved successfuly")
        else:
            print("Classifier object save failed")
