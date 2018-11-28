class MaximumEntropyClassifier:
    """
    implements MEMM, training and inference methods
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        """

        :param X: a HistoryTuple array of size (N,m)
        where N is the training set length, and m is the number of features
        (should wrap scipy?)
        :param y: a Tag array of size (N,)
        :return:
        """
        pass

    def predict(self, X):
        """

        :param X: a HistoryTuple array of size (N,m)
        :return: argmax of p(y|x;v)
        """
        pass

    def predict_probability(self, X):
        """

        :param X: a HistoryTuple array of size (N,m)
        :return: p(y|x;v)
        """
        pass