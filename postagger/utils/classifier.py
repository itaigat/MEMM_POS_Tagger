from postagger.utils.features import build_features, build_feature_matrix, compute_y_x_matrix
from postagger.utils.params import Params
from postagger.utils.common import poss
import numpy as np
from scipy.optimize import minimize
from postagger.utils.common import timeit
from time import time
import copy


class MaximumEntropyClassifier:
    """
    implements MEMM, training and inference methods
    """
    @timeit
    def __init__(self, iterable_sentences):
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
        print("Parsing iterables: %f s" % (time()-t1)); t2 = time()

        self.feature_matrix = build_feature_matrix(X, y, sentences, Params.features_fncs)
        print("Building feature matrix: %f s" % (time()-t2)); t3 = time()

        # compute y-x matrix (for each x in X , for each y in Y , vstack f(x,y))
        self.y_x_matrix = compute_y_x_matrix(X, sentences, Params.features_fncs)
        print("Building y_x features matrix: %f s" % (time() - t3))

        self.X = X
        self.y = y
        self.sentences = sentences
        self.reg = None
        # share between loss and grad
        self.normas = None
        self.scores = None

    @timeit
    def fit(self, reg=0, max_iter=1, max_fun=1):
        self.reg = reg
        m = self.feature_matrix.shape[1]
        v_init = np.zeros(m)

        # design note: minimize takes its own args (hence we pass loss, grad params without using self)
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
        fully vectorized MLE loss
        """
        loss = 0
        t1 = time()

        # fully vectorized computations
        first_term = feature_matrix.dot(v).sum()
        print("Loss first term: %f s" % (time()-t1)); t2 = time()

        second_term = self.compute_loss_second_term(v, X, sentences)
        print("Loss second term: %f s" % (time() - t2)); t3 = time()

        reg = (self.reg / 2) * np.sum(v**2)
        print("Loss reg: %f s" % (time() - t3)); t4 = time()

        # L(v) = a - b - regularization
        # recap goal: maximize L(v)
        loss = - (first_term - second_term - reg)
        print("Loss final sum: %f s" % (time() - t4))

        return loss

    def compute_loss_second_term(self, v, X, sentences):
        y_x_matrix = self.y_x_matrix  # shape (|Y|*|X|, m)
        dot_prod = y_x_matrix.dot(v)  # shape (|Y|*|X|, 1)
        dot_prod = dot_prod.reshape(-1, len(X))  # shape (|Y|, |X|)
        dot_prod_scores = np.exp(dot_prod)
        self.scores = copy.copy(dot_prod_scores)
        ret = np.sum(dot_prod_scores, axis=0)   # shape (|X|,)
        self.normas = copy.copy(ret)
        ret = np.log(ret)
        ret = np.sum(ret)
        return ret

    @timeit
    def grad(self, v, feature_matrix, X, y, sentences):
        """
        Fully vectorized grad computation
        """
        # for each entry in v we should compute the gradient
        grad = np.zeros_like(v)

        t1 = time()
        first_term = feature_matrix.sum(axis=0)
        print("Grad first term: %f s" % (time() - t1)); t2 = time()

        second_term = self.compute_grad_second_term(v, X, sentences)
        print("Grad second term: %f s" % (time() - t2)); t3 = time()

        # recap goal: maximize L(v), hence we use -grad
        grad = -(first_term - second_term)
        print("Grad last sum: %f s" % (time() - t3))

        grad = np.ravel(grad)

        return grad

    def compute_grad_second_term(self, v, X, sentences):
        y_x_matrix = self.y_x_matrix  # shape (|Y|*|X|, m)
        probs_matrix = self.scores / self.normas  # shape (|Y|, |X|)
        probs_matrix = probs_matrix.reshape(-1,1)  # shape (|Y|*|X|, 1)
        product = y_x_matrix.multiply(probs_matrix)  # shape (|Y|*X|, m)
        grad_features = product.sum(axis=0)  # shape (m,)
        return grad_features

    def predict_probability(self, X):
        """
        :param X: a HistoryTuple array of size (N,m)
        :return: p(y|x;v)
        """
        pass