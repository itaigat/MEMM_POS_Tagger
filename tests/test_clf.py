import numpy as np
from numpy.testing import assert_array_almost_equal


def test_first_loss(clf, feature_matrix, X, y, sentences):
    """
    test first loss evaluation on train_dev.wtag (unigram features)

    NOTICE: no regularization (lambda = 0)

    first test: clf produces feature matrix of size (29,45)

    second test: loss = 0 - 29*np.log(45) = -110.39321220333923
    """
    assert (feature_matrix.shape == (29, 45))

    m = feature_matrix.shape[1]
    v_init = np.zeros(m)
    # reg
    clf.reg = 0
    loss = clf.loss(v_init, feature_matrix, X, y, sentences)

    assert_array_almost_equal(loss, 110.39321220333923)


def test_first_loss_(clf, feature_matrix, X, y, sentences):
    # assumes train_dev_testclf and unigram features only
    assert (feature_matrix.shape == (2, 2))
    m = feature_matrix.shape[1]
    v_init = np.zeros(m)
    # reg
    clf.reg = 0
    loss = clf.loss(v_init, feature_matrix, X, y, sentences)

    assert_array_almost_equal(loss, 1.3862943611198906)


def test_first_grad_(clf, feature_matrix, X, y, sentences):
    # assumes train_dev_testclf and unigram features only
    m = feature_matrix.shape[1]
    v_init = np.zeros(m)
    clf.reg = 0
    out = clf.grad(v_init, feature_matrix, X, y, sentences)
    first_grad = [0, 0]  # (1-2*0.5, 1-2*0.5) , all get same probability when v == 0
    assert_array_almost_equal(out, first_grad)


def test_first_loss_grad_nonzero_v(clf, feature_matrix, X, y, sentences):
    # v_init = [0.5,1]
    m = feature_matrix.shape[1]
    v_init = np.ones(m)
    v_init[0] = 0.5
    # loss
    first_term = 0.5 + 1
    second_term = 2 * np.log(np.exp(0.5) + np.exp(1))
    first_loss = -(first_term - second_term)
    out = clf.loss(v_init, feature_matrix, X, y, sentences)
    assert_array_almost_equal(out, first_loss)
    # grad
    p0 = np.exp(0.5) / (np.exp(0.5) + np.exp(1))
    p1 = 1 - p0
    first_grad = [-(1 - 2 * p0), -(1 - 2 * p1)]
    out = clf.grad(v_init, feature_matrix, X, y, sentences)
    assert_array_almost_equal(out, first_grad)


def test_first_grad(clf, feature_matrix, X, y, sentences):
    """
    test first grad evaluation on train_dev.wtag (unigram features)

    basically the grad is composed of two terms, grad = A - B , vector of shape (m,)
    where A is the number of times the specific feature was enabled (on all samples), what's called empirical counts
    and B is the expected counts (according to the current v) for that feature, over all samples

    for example, the first tag (CC) wasn't seen on training data, then its A=0
    its B=sum_i (1/45) = 29/45 , => grad_CC = 0 - 29/45
    """
    m = feature_matrix.shape[1]
    v_init = np.zeros(m)
    # reg
    clf.reg = 0
    out = clf.grad(v_init, feature_matrix, X, y, sentences)
    first_grad = [-0.64444444, 1.35555556, 3.35555556, - 0.64444444, - 0.64444444, 1.35555556,
                  2.35555556, - 0.64444444, 0.35555556, - 0.64444444, - 0.64444444, 3.35555556,
                  1.35555556, 1.35555556, - 0.64444444, - 0.64444444, - 0.64444444, - 0.64444444,
                  - 0.64444444, - 0.64444444, 0.35555556, - 0.64444444, - 0.64444444, - 0.64444444,
                  0.35555556, - 0.64444444, - 0.64444444, 1.35555556, - 0.64444444, 0.35555556,
                  - 0.64444444, - 0.64444444, - 0.64444444, - 0.64444444, - 0.64444444, - 0.64444444,
                  - 0.64444444, - 0.64444444, - 0.64444444, - 0.64444444, - 0.64444444, - 0.64444444,
                  1.35555556, 1.35555556, - 0.64444444]
    first_grad = np.array(first_grad)
    first_grad = -1 * first_grad

    # up to 6 decimal dots
    assert_array_almost_equal(out, first_grad)


def run_clf_tests(clf, feature_matrix, X, y, sentences):
    """
    run all classifier tests
    """
    # test_first_loss_(clf, feature_matrix, X, y, sentences)
    # test_first_grad_(clf, feature_matrix, X, y, sentences)
    test_first_loss_grad_nonzero_v(clf, feature_matrix, X, y, sentences)
