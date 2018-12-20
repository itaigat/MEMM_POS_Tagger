import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from postagger.utils.common import timeit
from time import time
import copy
import operator
from postagger.utils.features import build_y_x_matrix, build_feature_matrix_, init_callable_features
from postagger.utils.common import poss
from postagger.utils.params import Params
from postagger.utils.common import viterbi_s
from postagger.utils.common import pickle_save, pickle_load
from postagger.utils.score import accuracy, confusion_matrix

epsilon = 1e-32  # for numeric issues


class MaximumEntropyClassifier:
    """
    implements MEMM, training and inference methods
    """

    def __init__(self, iterable_sentences, preprocess_dict, common_words):
        """
        iterable sentences:
            a tuple (tuples, tags, stripped_sentence)

            tuples: [('*', '*', 0, 0), ('*', 'DT', 0, 1),..]
            tags: ['DT', 'NNP',...]
            stripped_sentence: ['All', 'Nasdaq', ..]

            tuples are history tuples <u,v,sentence,i> ,
            where sentence is an id, refers to its position on the corpus, e.g., first sentence is 0.
        """
        # prepare for converting x,y (history-tuple and tag)
        # into features matrix
        t1 = time()
        X, y, sentences = [], [], []
        for tuples, tags, sentence in iterable_sentences:
            for i in range(len(tuples)):
                X.append(tuples[i])
                y.append(tags[i])
            sentences.append(sentence)
        print("Parsing iterables: %f s" % (time() - t1))
        t2 = time()

        self.callable_functions = init_callable_features(poss, Params, preprocess_dict, common_words)

        # build matrices
        self.feature_matrix = build_feature_matrix_(X, y, sentences, self.callable_functions)
        print("Building feature matrix: %f s" % (time() - t2))
        t3 = time()

        self.y_x_matrix = build_y_x_matrix(X, poss, sentences, self.callable_functions)
        print("Building y_x features matrix: %f s" % (time() - t3))

        self.X = X
        self.y = y
        self.sentences = sentences
        self.reg = 0
        # share between loss and grad
        self.normas = None
        self.scores = None
        self.v = None

    def fit(self, reg=0, verbose=0):
        self.reg = reg
        m = self.feature_matrix.shape[1]
        v_init = np.zeros(m)

        # design note: minimize takes its own args (hence we pass loss, grad params without using self)
        # res = (self.loss, v_init, method='L-BFGS-B', jac=self.grad,
        #                args=(self.feature_matrix, self.X, self.y, self.sentences),
        #                options={'disp': verbose, 'maxiter': 30})

        res = fmin_l_bfgs_b(func=self.loss, x0=v_init, fprime=self.grad,
                            args=(self.feature_matrix, self.X, self.y, self.sentences))

        # save normalized parameters vector (normalization is needed as numeric fix for viterbi computations)
        norma = np.linalg.norm(res[0], ord=1)
        self.v = res[0] / res[0].sum()

        print('Weights vector shape: ', res[0].shape)
        print('Weights vector: ', res[0])

    def loss(self, v, feature_matrix, X, y, sentences):
        """
        MLE loss
        """
        loss = 0
        t1 = time()

        # fully vectorized computations
        first_term = feature_matrix.dot(v).sum()
        print("Loss first term: %f s" % (time() - t1))
        t2 = time()

        second_term = self.compute_loss_second_term(v, X, sentences)
        print("Loss second term: %f s" % (time() - t2))
        t3 = time()

        reg = (self.reg / 2) * np.sum(v ** 2)
        print("Loss reg: %f s" % (time() - t3))
        t4 = time()

        # L(v) = a - b - regularization
        # recap goal: maximize L(v)
        loss = - (first_term - second_term - reg)
        print("Loss final sum: %f s" % (time() - t4))
        print("Loss: %f" % loss)
        return loss

    def compute_loss_second_term(self, v, X, sentences):
        """helper"""
        # y_x_sum_rows = self.y_x_matrix.sum(axis=1)
        # enabled = y_x_sum_rows.reshape(len(self.X), len(poss)).sum(axis=0)
        # tag_weights = (enabled / enabled.sum()).T
        y_x_matrix = self.y_x_matrix  # shape (|Y|*|X|, m)
        y_x_sum_rows = self.y_x_matrix.sum(axis=1)
        dot_prod = y_x_matrix.dot(v)  # shape (|Y|*|X|, 1)
        dot_prod = dot_prod.reshape(len(poss), -1)  # shape (|Y|, |X|)
        # fixes numeric issues, break tests
        max_point = dot_prod.max()
        dot_prod = dot_prod - max_point
        dot_prod_scores = np.exp(dot_prod)
        self.scores = copy.copy(dot_prod_scores) # shape (|Y|, |X|)
        ret = np.sum(dot_prod_scores, axis=0)  # shape (|X|,)
        self.normas = copy.copy(ret)
        ret = np.log(ret + epsilon)  # numeric issues
        ret = np.sum(ret)
        return ret

    def grad(self, v, feature_matrix, X, y, sentences):
        """
        Fully vectorized grad computation
        """
        # for each entry in v we should compute the gradient
        grad = np.zeros_like(v)

        t1 = time()
        first_term = feature_matrix.sum(axis=0)
        print("Grad first term: %f s" % (time() - t1))
        t2 = time()

        second_term = self.compute_grad_second_term(v, X, sentences)
        print("Grad second term: %f s" % (time() - t2))
        t3 = time()

        reg_grad = v

        # recap goal: maximize L(v), hence we use -grad
        grad = -(first_term - second_term + reg_grad)
        print("Grad last sum: %f s" % (time() - t3))

        grad = np.ravel(grad)
        print("Grad sum: %f" % np.sum(grad**2))
        return grad

    def compute_grad_second_term(self, v, X, sentences):
        """helper"""
        y_x_matrix = self.y_x_matrix  # shape (|Y|*|X|, m)
        # probs_matrix = self.scores / (self.normas.reshape
        #                               (-1, 1) + epsilon)  # shape (|Y|, |X|) # fix numeric issue
        probs_matrix = self.scores / (self.normas + epsilon)
        probs_matrix = probs_matrix.reshape(-1, 1)  # shape (|Y|*|X|, 1)
        product = y_x_matrix.multiply(probs_matrix)  # shape (|Y|*X|, m)
        grad_features = product.sum(axis=0)  # shape (m,)
        return grad_features

    def predict_probability(self, X):
        """
        :param x: a tuple (t, u, sentence_id, word_id)
        :return: q(v|t,u,sentece_id,word_id)
        """

        pass

    def calc_q(self, X):
        pass

    def predict(self, X):
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
                tmp = viterbi_s(tuples[0][2], sentences, self.v, self.callable_functions)
                tags_predicted.append(tmp)
                true_tags.append(tags)
                print("Sentence: " + str(sentence))
                print("Tags: " + str(tags))
                print("Pred: " + str(tmp))

        print('Accuracy: ', accuracy(tags_predicted, true_tags))
        print('Confusion Matrix:')
        print(confusion_matrix(tags_predicted, true_tags))

        return tags_predicted

    def get_num_features(self):
        return self.feature_matrix.shape[1]

    def get_enabled_features_per_tag(self):
        y_x_sum_rows = self.y_x_matrix.sum(axis=1)
        enabled = y_x_sum_rows.reshape(len(self.X), len(poss)).sum(axis=0)
        counts = enabled.tolist()[0]
        tuples_list = []
        for i,count in enumerate(counts):
            tup = (poss[i], count)
            tuples_list.append(tup)

        counts_dict = dict(tuples_list)

        sorted_dict = sorted(counts_dict.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_dict


def save_load_init_model(initialized_clf, filename):
    """saves or loads clf model
    initialized_clf: classifier object after running constructor
    filename: if empty then function assumes user wants to load"""
    # try loading
    clf = None
    if initialized_clf is None:
        clf = pickle_load(filename)
    if clf is not None:
        print("Loaded classifier object")
        return clf

    # save
    ret = None
    ret = pickle_save(initialized_clf, filename)
    if ret:
        print("Classifier object saved successfuly")
    else:
        print("Classifier object save failed")
