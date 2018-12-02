from src.utils.features import build_features
from src.utils.params import Params
from src.utils.common import poss
import numpy as np
from scipy.optimize import minimize

LAMBDA = 0

class MaximumEntropyClassifier:
    """
    implements MEMM, training and inference methods
    """
    def __init__(self):
        pass

    def fit(self, iterable_sentences):
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
        X, y, sentences = [], [], []
        for tuples, tags, sentence in iterable_sentences:
            for i in range(len(tuples)):
                X.append(tuples[i])
                y.append(tags[i])
            sentences.append(sentence)

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

        # next we should feed both
        #feature_matrix, np.array(y)
        X = feature_matrix
        y = np.array(y)
        m = feature_matrix.shape[1]
        v_init = np.zeros(m)

        res = minimize(self.loss, v_init, method='L-BFGS-B', jac=self.jacob, args=(X, y, sentences))

        if res.success:
            print("Optimization succeeded.")

        print(res.x)
        print(res.x.shape)

    def loss(self, v, X, y, sentences):
        """
        defines softmax loss
        :param X:
        :param y:
        :return: -log-likelihood
        """
        m = X.shape[1]
        loss = np.sum(v.dot(X.T))  # vectorized

        second_term = 0
        # for each history tuple compute unnormlalized log prob
        for i,x in enumerate(X):
            second_term += self.compute_y_matrix(v, x, sentences)

        reg = (-LAMBDA / 2) * np.sum(v*v)

        loss -= (second_term + reg)

        return np.zeros(m)

    def compute_y_matrix(self, v, x, sentences):
        """
        iterates over all y's for a given X
        and compute log sum_y (e ^ (v * f(x,y))

        practiaclly build f(x,y) for each y, then use broadcasting to compute v * f(x,y)
        :return:
        """
        epsilon = 1e-16
        # need to iterate over all possible y's
        # build its matrix iteratively
        feature_matrix = None
        for i, pos in enumerate(poss):
            f = build_features(x, poss[i], sentences, Params.features_fncs)  # f shape: (m,)
            if i == 0:
                feature_matrix = np.array(f)  # first row, init as array
            else:
                feature_matrix = np.vstack((feature_matrix, np.array(f)))  # add another row to matrix

        ret = np.log(np.sum(v.dot(feature_matrix.T)) + epsilon)

        return ret

        # finally ret = log ( np.sum ( v * features_matrix ) )

    def jacob(self, v, X, y, sentences):
        """
        defines
        :param X:
        :param y:
        :return:
        """
        #loss = x*V
        m = X.shape[1]

        #  TODO: compute gradient
        return np.zeros(m)

    def predict_probability(self, X):
        """

        :param X: a HistoryTuple array of size (N,m)
        :return: p(y|x;v)
        """
        pass
