import sys
from postagger.utils.classifier import MaximumEntropyClassifier
from postagger.utils.common import timeit, get_data_path
from postagger.utils.common import get_tags
from postagger.utils.preprocess import load_save_preprocessed_data
from postagger.utils.decoder import CompData
from postagger.utils.classifier import save_load_init_model

# params
load_model = True
load_matrices = False
load_preprocess = True
model_name = 'model_dev.pickle'
model_matrices = 'model_matrices_dev.pickle'
model_preprocess = 'model_preprocess_dev.pickle'
verbose = 1

# data files
train = 'train.wtag'
test = 'dev/test_dev.wtag'
comp = 'comp.words'


# hyper params

# features
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

# model
regularization = 1

@timeit
def main(argv):



    if load_model:
        clf = save_load_init_model(clf=None, filename=model_name)
    else:
        if load_matrices:
            clf = save_load_init_model(clf=None, filename=model_matrices)

        else:
            # train
            train_path = get_data_path(train)
            train_sentences = CompData(train_path)

            # count features occurrences
            preprocessor = load_save_preprocessed_data(model_preprocess, train_sentences, load=load_preprocess)
            # apply filtering
            pdict = preprocessor.summarize_counts(method='cut', dict=min_occurrence_dict)
            # init classifier with known tags
            tags = get_tags(train)
            clf = MaximumEntropyClassifier(train_sentences, pdict, tags)
            save_load_init_model(clf=clf, filename=model_matrices)

        print("Start fitting %d features" % clf.get_num_features())
        print("Top enabled features per tag: " + str(clf.get_enabled_features_per_tag()))
        clf.fit(reg=regularization, verbose=verbose)
        save_load_init_model(clf=clf, filename=model_name)

    # evaluate
    test_path = get_data_path(test)
    test_sentences = CompData(test_path)
    t_predict = clf.predict(test_sentences)
    print(t_predict)

    """
    comp_path = get_data_path(comp)
    comp_sentences = CompData(comp_path, comp=True)
    comp_predict = clf.predict(comp_sentences)
    print(comp_predict)
    """


if __name__ == '__main__':
    main(sys.argv[1:])
