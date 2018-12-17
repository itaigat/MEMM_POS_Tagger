import numpy as np
from scipy.optimize import minimize
from postagger.utils.common import timeit
from time import time
import copy
from postagger.utils.features import build_y_x_matrix, build_feature_matrix_, init_callable_features
from postagger.utils.common import poss
from postagger.utils.params import Params
from postagger.utils.common import viterbi
from postagger.utils.common import pickle_save, pickle_load

epsilon = 1e-32  # for numeric issues

class MaximumEntropyClassifier:
    """
    implements MEMM, training and inference methods
    """

    @timeit
    def __init__(self, iterable_sentences, preprocess_dict):
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

        self.callable_functions = init_callable_features(poss, Params, preprocess_dict)

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

    @timeit
    def fit(self, reg=0, verbose=0):
        self.reg = reg
        m = self.feature_matrix.shape[1]
        v_init = np.zeros(m)

        # design note: minimize takes its own args (hence we pass loss, grad params without using self)
        res = minimize(self.loss, v_init, method='L-BFGS-B', jac=self.grad,
                       args=(self.feature_matrix, self.X, self.y, self.sentences),
                       options={'disp': verbose})

        # save normalized parameters vector (normalization is needed as numeric fix for viterbi computations)
        norma = np.linalg.norm(res.x, ord=1)
        self.v = res.x / norma

        print("Optimization succeeded.")
        print('Weights vector shape: ', res.x.shape)
        print('Weights vector: ', res.x)

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
        print(loss)
        return loss

    def compute_loss_second_term(self, v, X, sentences):
        """helper"""
        y_x_matrix = self.y_x_matrix  # shape (|Y|*|X|, m)
        dot_prod = y_x_matrix.dot(v)  # shape (|Y|*|X|, 1)
        dot_prod = dot_prod.reshape(-1, len(X))  # shape (|Y|, |X|)
        # fixes numeric issues, break tests
        max_point = dot_prod.max()
        dot_prod = dot_prod - max_point
        dot_prod_scores = np.exp(dot_prod)
        self.scores = copy.copy(dot_prod_scores)
        ret = np.sum(dot_prod_scores, axis=1)  # shape (|X|,)
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

        reg_grad = np.sum(v)

        # recap goal: maximize L(v), hence we use -grad
        grad = -(first_term - second_term + reg_grad)
        print("Grad last sum: %f s" % (time() - t3))

        grad = np.ravel(grad)

        return grad

    def compute_grad_second_term(self, v, X, sentences):
        """helper"""
        y_x_matrix = self.y_x_matrix  # shape (|Y|*|X|, m)
        probs_matrix = self.scores / (self.normas.reshape(-1, 1) + epsilon)  # shape (|Y|, |X|) # fix numeric issue
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

    @timeit
    def predict(self, X):
        """
        :param X: [(sentence_index, sentence_list),...,()]
        :return: [tags_sentence_1...]
        """
        tags_predicted = []
        sentences = []

        for tuples, tags, sentence in X:
            sentences.append(sentence)

        if self.v is None:
            print('You need to fit the model before prediction')
        else:
            for tuples, tags, sentence in X:
                tmp = viterbi(tuples[0][2], sentences, self.v, self.callable_functions)
                tags_predicted.append(tmp)
                print("Sentence: " + str(sentence))
                print("Tags: " + str(tags))
                print("Pred: " + str(tmp))

        return tags_predicted


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

    # save and return loaded
    ret = None
    ret = pickle_save(initialized_clf, filename)
    if ret:
        print("Classifier object saved successfuly")
    else:
        print("Classifier object save failed")
