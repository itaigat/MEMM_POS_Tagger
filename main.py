import sys

from postagger.utils.classifier import MaximumEntropyClassifier
from postagger.utils.common import timeit, get_data_path
from postagger.utils.preprocess import load_save_preprocessed_data
from postagger.utils.decoder import CompData
from postagger.utils.classifier import save_load_init_model


#  TODO:
#   perform sanity check for viterbi to be sure it is not biased

#  TODO:
#   add TOP X counting method,
#   build as external method: use preprocess_dict.pdict attribute as input which is a dict of lists
#   each value is a list that contains the corresponding counts computed from data, see example below:
#   preprocess_dict.pdict = { 'wordtag-f100': [('the', 'DT'), ('dog', 'NN'), ('the', 'dt')...], 'suffix-f101':[ ...].. }
#   NOTICE: don't change Preprocess class, otherwise train.pickle file won't load and you'll have to build it again

#  TODO:
#   other option, since the feature we choose from counts are somehow biased to 'DT', 'NN', 'IN', suggest a counting
#   method that prefers variety of tags.
#   both this and previous method should solve the biased results of the clf.

#  TODO:
#   accuracy,
#   confusion matrix top10 mistakes,
#   add arguments from main sys.argv (to run on server),
#   add final model saver (after optimization)


min_occurrence_dict = {
    'wordtag-f100': 50,
    'suffix-f101': 250,
    'prefix-f102': 250,
    'trigram-f103': 50,
    'bigram-f104': 50,
    'unigram-f105': 50,
    'previousword-f106': 100,
    'nextword-f107': 100,
    'starting_capital': 5,
    'capital_inside': 5,
    'number_inside': 5
}


@timeit
def main(argv):
    # args
    # clf saved is not optimized, only matrices
    load_matrices_from_disk = False  # False means clf object will be saved
    init_clf_filename = 'init_clf.pickle'

    # training procedure
    train_path = get_data_path('train.wtag')
    train_sentences = CompData(train_path)
    # count features occurrences
    preprocess_dict = load_save_preprocessed_data('train.pickle', train_sentences)
    pdict_cut = preprocess_dict.cut(min_occurrence_dict)
    # build matrices
    if load_matrices_from_disk:
        # load
        clf = save_load_init_model(initialized_clf=None, filename=init_clf_filename)
    else:
        # save
        clf = MaximumEntropyClassifier(train_sentences, pdict_cut)
        save_load_init_model(initialized_clf=clf, filename=init_clf_filename)

    # fit
    clf.fit(reg=10, verbose=1)

    # evaluate
    test_path = get_data_path('test.wtag')
    test_sentences = CompData(test_path)
    t_predict = clf.predict(test_sentences)
    print(t_predict)

    """
    comp_path = get_data_path('comp.words')
    comp_sentences = CompData(comp_path, comp=True)
    comp_predict = clf.predict(comp_sentences)
    print(comp_predict)
    """
    # tests (pass on train_dev and unigram features only): CURRENTLY BREAKS (fix of numeric issues)
    # run_clf_tests(clf, clf.feature_matrix, clf.X, clf.y, clf.sentences)
    # clf.fit(reg=10, verbose=1)


if __name__ == '__main__':
    main(sys.argv[1:])
