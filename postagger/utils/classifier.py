from postagger.utils.features import build_features
from postagger.utils.params import Params
from postagger.utils.common import poss
import numpy as np
from scipy.optimize import minimize
from postagger.utils.common import timeit
from time import time

LAMBDA = 0

class MaximumEntropyClassifier:
    """
    implements MEMM, training and inference methods
    """
    @timeit
    def __init__(self, iterable_sentences):
        # prepare for converting x,y (history-tuple and tag)
        # into features matrix
        t1 = time()
        X, y, sentences = [], [], []
        for tuples, tags, sentence in iterable_sentences:
            for i in range(len(tuples)):
                X.append(tuples[i])
                y.append(tags[i])
            sentences.append(sentence)
        print("Parsing iterables: %f s" % (time()-t1)); t2 = time()

        # build features matrix
        feature_matrix = None
        len_dataset = len(X)

        for i in range(len_dataset):
            f = build_features(X[i], y[i], sentences, Params.features_fncs)  # f shape: (m,)
            if i == 0:
                feature_matrix = np.array(f)  # first row, init as array
            else:
                feature_matrix = np.vstack((feature_matrix, np.array(f)))  # add another row to matrix

        feature_matrix = np.array(feature_matrix)
        print("Building feature matrix: %f s" % (time()-t2)); t3 = time()

        # compute y-x matrix (for each x in X , for each y in Y , vstack f(x,y))
        self.y_x_matrix = self.compute_y_x_matrix(X, sentences)
        print("Building y_x features matrix: %f s" % (time() - t3))


        self.feature_matrix = feature_matrix
        self.X = X
        self.y = np.array(y)
        self.sentences = sentences

    @timeit
    def fit(self, reg=0, max_iter=1, max_fun=1):
        """
        iterable sentences:
            a tuple (tuples, tags, stripped_sentence)

            tuples: [('*', '*', 0, 0), ('*', 'DT', 0, 1),..]
            tags: ['DT', 'NNP',...]
            stripped_sentence: ['All', 'Nasdaq', ..]

            tuples are history tuples <u,v,sentence,i> ,
            where sentence is an id, refers to its position on the corpus, e.g., first sentence is 0.
        """
        m = self.feature_matrix.shape[1]
        v_init = np.zeros(m)

        res = minimize(self.loss, v_init, method='L-BFGS-B', jac=self.grad,
                       args=(self.feature_matrix, self.X, self.y, self.sentences),
                       options={'disp': 0, 'maxiter': max_iter, 'maxfun': max_fun})

        if res.success:
            print("Optimization succeeded.")

        print(res.x)
        print(res.x.shape)

    @timeit
    def loss(self, v, feature_matrix, X, y, sentences):
        """
        defines softmax loss
        :param X:
        :param y:
        :return: -log-likelihood
        """
        loss = 0
        t1 = time()

        # fully vectorized computations
        first_term = np.sum(v.dot(feature_matrix.T))
        print("Loss first term: %f s" % (time()-t1)); t2 = time()

        second_term = 0
        for i, x in enumerate(feature_matrix):
            second_term += self.compute_normalization(v,  self.X[i], sentences)  # TODO: fully vectorized op
        print("Loss second term: %f s" % (time() - t2)); t3 = time()

        reg = (LAMBDA / 2) * np.sum(v**2)
        print("Loss reg: %f s" % (time() - t3)); t4 = time()

        # L(v) = a - b - regularization
        # recap goal: maximize L(v)
        loss = - (first_term - second_term - reg)
        print("Loss final sum: %f s" % (time() - t4))

        return loss

    def compute_normalization(self, v, x, sentences):
        """
        iterates over all y's for a given x
        practiaclly build f(x,y) for each y, then use broadcasting to compute v * f(x,y)

        :return: log sum_y (e ^ (v * f(x,y))
        """
        y_matrix = self.compute_y_matrix(x, sentences)

        ret = np.log(np.sum(np.exp(v.dot(y_matrix))))

        return ret

    def compute_y_x_matrix(self, X, sentences):
        """
        for each x in X, compute all y's for x:
        works by vertical stacking of compute_y_matrix which is shape (|Y|,m)
        output of shape (|Y|*|X|,m)
        """
        for i, x in enumerate(X):
            f = self.compute_y_matrix(x, sentences)  # f shape: (m,)
            if i == 0:
                feature_matrix = np.array(f)  # first row, init as array
            else:
                feature_matrix = np.vstack((feature_matrix, np.array(f)))  # add another row to matrix

        return feature_matrix

    def compute_y_matrix(self, x, sentences):
        """
        helper
        need to iterate over all possible y's for a given x
        build its matrix iteratively

        output of shape (tags, features)
        """
        feature_matrix = None
        for i, pos in enumerate(poss):
            f = build_features(x, poss[i], sentences, Params.features_fncs)  # f shape: (m,)
            if i == 0:
                feature_matrix = np.array(f)  # first row, init as array
            else:
                feature_matrix = np.vstack((feature_matrix, np.array(f)))  # add another row to matrix

        return feature_matrix

    @timeit
    def grad(self, v, feature_matrix, X, y, sentences):
        """
        defines
        :param X:
        :param y:
        :return:
        """
        # for each entry in v we should compute the gradient
        grad = np.zeros_like(v)

        t1 = time()
        first_term = np.sum(self.feature_matrix.T, axis=1)
        print("Grad first term: %f s" % (time() - t1)); t2 = time()

        second_term = np.zeros_like(v)
        for i, x in enumerate(self.feature_matrix):  # TODO: fully vectorized op
            y_matrix = self.compute_y_matrix(x, sentences)  # shape (tags, features)
            probs_vec = self.predict_all_ys(v, x, y_matrix)
            second_term += (y_matrix.T).dot(probs_vec)
        print("Grad second term: %f s" % (time() - t2)); t3 = time()

        # recap goal: maximize L(v), hence we use -grad
        grad = -(first_term - second_term)
        print("Grad last sum: %f s" % (time() - t3))

        return grad

    def predict_all_ys(self, v, x, y_matrix):
        """
        predict probability for a given x over all y's
        :param x:
        :return: vector of probabilities for each tag
        """
        numerator = np.exp(v.dot(y_matrix.T))
        denom = np.sum(numerator)

        ret = numerator / denom
        # in first iteration (v == 0) all y's should have same prob (1/|tags| = 1/45)
        return ret

    def predict_probability(self, X):
        """
        :param X: a HistoryTuple array of size (N,m)
        :return: p(y|x;v)
        """
        pass