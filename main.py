import sys

from postagger.utils.classifier import MaximumEntropyClassifier
from postagger.utils.common import timeit, get_data_path
from postagger.utils.preprocess import load_save_preprocessed_data
from postagger.utils.decoder import CompData
from postagger.utils.classifier import save_load_init_model


#  TODO:
#   improve viterbi efficiency

#  TODO:
#   confusion matrix top10 mistakes,
#   add arguments from main sys.argv (to run on server),
#   add final model saver (after optimization)


min_occurrence_dict = {
    'wordtag-f100': 0,
    'suffix-f101': 0,
    'prefix-f102': 0,
    'trigram-f103': 0,
    'bigram-f104': 0,
    'unigram-f105': 0,
    'previousword-f106': 0,
    'nextword-f107': 0,
    'starting_capital': 0,
    'capital_inside': 0,
    'number_inside': 0,
    'hyphen_inside': 0,
    'pre_pre_word': 0,
    'next_next_word': 0
}

top_occurrence = {
    'wordtag-f100': 50,
    'suffix-f101': 50,
    'prefix-f102': 5,  # junk
    'trigram-f103': 780,
    'bigram-f104': 100,
    'unigram-f105': 50,
    'previousword-f106': 320,
    'nextword-f107': 307,
    'starting_capital': 10,
    'capital_inside': 3,
    'number_inside': 3,
    'hyphen_inside': 100,  # unknown
    'pre_pre_word': 3,  # unknown
    'next_next_word': 3  # unknown
}

top_common_words = 100

@timeit
def main(argv):
    # args
    # only matrices are saved, not the optimized clf
    load_matrices_from_disk = False  # False means clf object will be saved, enable to debug clf.fit / predict
    init_clf_filename = 'init_clf_nolimit_common_extra_features.pickle'  # change when training a new model

    if load_matrices_from_disk:
        clf = save_load_init_model(initialized_clf=None, filename=init_clf_filename)
    else:
        # build matrices and save to disk
        train_path = get_data_path('train.wtag')
        train_sentences = CompData(train_path)

        # count features occurrences
        preprocessor = load_save_preprocessed_data('train_preprocessed_common_extra_features.pickle', train_sentences,
                                                   top_words=top_common_words, load=True)
        pdict = preprocessor.summarize_counts(method='cut', dict=min_occurrence_dict)

        clf = MaximumEntropyClassifier(train_sentences, pdict)
        save_load_init_model(initialized_clf=clf, filename=init_clf_filename)

    # fit
    print("Training %d features" % clf.get_num_features())
    print("Top enabled features per tag: " + str(clf.get_enabled_features_per_tag()))

    clf.fit(reg=0, verbose=1)

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
