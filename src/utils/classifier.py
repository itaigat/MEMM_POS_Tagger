import numpy as np
from scipy.optimize import minimize

from src.utils.common import poss
from src.utils.features import build_features, get_feature_matrix
from src.utils.params import Params


class MaximumEntropyClassifier:
    """
    implements MEMM, training and inference methods
    """

    def __init__(self):
        self.LAMBDA = 0

        self.X = None
        self.y = None

    def fit(self, iterable_sentences, feature_funcs=Params.features_funcs):
        """
        iterable sentences:
            a tuple (tuples, tags, stripped_sentence)

            tuples: [('*', '*', 0, 0), ('*', 'DT', 0, 1),..]
            tags: ['DT', 'NNP',...]
            stripped_sentence: ['All', 'Nasdaq', ..]

            tuples are history tuples <u,v,sentence,i> ,
            where sentence is an id, refers to its position on the corpus, e.g., first sentence is 0.
        """

        y_tmp = iterable_sentences.get_tags()
        sentences = iterable_sentences.get_sentences()

        self.X = get_feature_matrix(iterable_sentences, feature_funcs)
        self.y = np.array(y_tmp)

        m = self.X.shape[1]
        v_init = np.zeros(m)

        res = minimize(self.loss, v_init, method='L-BFGS-B', jac=self.grad, args=(self.X, self.y, sentences))

        if res.success:
            print("Optimization succeeded.")

        print(res.x)
        print(res.x.shape)

    def loss(self, v, sentences):
        """
        Computes softmax loss
        :param v:
        :param sentences:
        :return: -log-likelihood
        """

        # fully vectorized computations
        first_term = np.sum(v.dot(self.X.transpose()))  # TODO: Check if it works
        second_term = 0
        for i, x in enumerate(self.X):
            second_term += self.compute_normalization(v, x, sentences)

        reg = (-self.LAMBDA / 2) * np.sum(v ** 2)

        # L(v) = a - b - regularization
        loss = first_term - second_term - reg

        return loss

    @staticmethod
    def compute_normalization(v, x, sentences):
        """
        iterates over all y's for a given X
        and compute log sum_y (e ^ (v * f(x,y))

        practically build f(x,y) for each y, then use broadcasting to compute v * f(x,y)

        do for each x in X and sum
        :return:
        """
        # need to iterate over all possible y's
        # build its matrix iteratively
        feature_matrix = None

        for i, pos in enumerate(poss):
            f = build_features(x, poss[i], sentences, Params.features_funcs)  # f shape: (m,)
            if i == 0:
                feature_matrix = np.array(f)  # first row, init as array
            else:
                feature_matrix = np.vstack((feature_matrix, np.array(f)))  # add another row to matrix

        ret = np.log(np.sum(np.exp(v.dot(feature_matrix.transpose()))))

        return ret

    def grad(self, v):
        """
        defines
        :param v:
        :return:
        """
        # loss = x * V
        m = self.X.shape[1]
        # for each entry in v we should compute the gradient
        grad = np.zeros_like(v)
        first_term = np.sum(self.X.transpose(), axis=0)

        #  TODO: compute gradient

        return np.zeros(m)

    def predict_probability(self, X_test):
        """
        :param X_test: a HistoryTuple array of size (N,m)
        :return: p(y|x;v)
        """
        pass
