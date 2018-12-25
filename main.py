import sys
from postagger.utils.classifier import MaximumEntropyClassifier
from postagger.utils.common import timeit, get_data_path
from postagger.utils.common import get_tags
from postagger.utils.preprocess import load_save_preprocessed_data
from postagger.utils.decoder import CompData
from postagger.utils.classifier import save_load_init_model

# params
load_model = False
load_matrices = False
load_preprocess = False
model_name = 'model2.pickle'
model_matrices = 'model2_matrices.pickle'
model_preprocess = 'model2_preprocess.pickle'
verbose = 1

# data files
train = 'train.wtag'
test = 'train2.wtag'
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
    'previousword-f106': 2,
    'nextword-f107': 2,
    'starting_capital': 0,
    'capital_inside': 0,
    'number_inside': 0,
    'hyphen_inside': 0,
    'pre_pre_word': 2,
    'next_next_word': 2
}

# model
regularization = 1


@timeit
def main():
    train_path = get_data_path(train)
    train_sentences = CompData(train_path)

    if load_model:
        clf = save_load_init_model(clf=None, filename=model_name)
    else:
        if load_matrices:
            clf = save_load_init_model(clf=None, filename=model_matrices)
        else:
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
    # train
    print("Evaluate train:")
    train_predict = clf.predict(train_sentences)
    print(train_predict)

    # test
    print("Evaluate test:")
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


def training():
    train_path = get_data_path(train)
    train_sentences = CompData(train_path)

    test_path = get_data_path(test)
    test_sentences = CompData(test_path)

    preprocessor = load_save_preprocessed_data(model_preprocess, train_sentences, load=load_preprocess)
    # apply filtering
    pdict = preprocessor.summarize_counts(method='cut', dict=min_occurrence_dict)
    # init classifier with known tags
    tags = get_tags(train)
    clf = MaximumEntropyClassifier(train_sentences, pdict, tags)

    reg = [5e-3, 1e-2, 5e-2, 1e-1, 1, 3, 5, 10, 25, 50, 100, 500, 1000]
    best_model = 'best_model.pickle'
    best_acc = 0
    test_acc = 0
    results = {}
    for r in reg:
        print("Start fitting model, reg: ", str(r))
        clf.fit(reg=r)
        try:
            print("Evaluate train:")
            train_pred = clf.predict(train_sentences)
            train_acc = train_pred['accuracy']
            print("Evaluate test:")
            test_pred = clf.predict(test_sentences)
            test_acc = test_pred['accuracy']
            results[('reg', r)] = {'train_acc': train_acc, 'test_acc': test_acc}
            if test_acc > best_acc:
                best_acc = test_acc
                save_load_init_model(clf=clf, filename=best_model)
        except:
            pass
        print("Current results", results)
        print("\n\n")

    print("Final results")
    print(results)


def training2():
    # data files
    train = 'train2.wtag'

    train_path = get_data_path(train)
    train_sentences = CompData(train_path, slice=(0, 5))

    validation_sentences = CompData(train_path, slice=(5, 6))

    preprocessor = load_save_preprocessed_data(model_preprocess, train_sentences, load=load_preprocess)

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

    reg = [1e-3, 1e-2, 1e-1]
    min_occur = [ {'previousword-f106': 1, 'nextword-f107':1, 'pre_pre_word': 0, 'next_next_word': 0},
                  {'previousword-f106': 1, 'nextword-f107':1, 'pre_pre_word': 1, 'next_next_word': 1},
                  {'previousword-f106': 1, 'nextword-f107': 1, 'pre_pre_word': 2, 'next_next_word': 2},
                  {'previousword-f106': 1, 'nextword-f107': 1, 'pre_pre_word': 1, 'next_next_word': 1},
                  {'previousword-f106': 0, 'nextword-f107': 0, 'pre_pre_word': 2, 'next_next_word': 2},
                  {'previousword-f106': 2, 'nextword-f107': 2, 'pre_pre_word': 0, 'next_next_word': 0}]
    best_model = 'best_model2.pickle'
    best_acc = 0
    test_acc = 0
    results = {}

    for occur_dict in min_occur:
        # apply filtering
        # update dict
        print("Init new classifier with updated occurrence dict")
        print(occur_dict)
        for key, value in occur_dict.items():
            min_occurrence_dict[key] = value
        pdict = preprocessor.summarize_counts(method='cut', dict=min_occurrence_dict)
        # init classifier with known tags
        tags = get_tags(train)
        clf = MaximumEntropyClassifier(train_sentences, pdict, tags)
        for r in reg:

            print("Start fitting model, reg: ", str(r))
            clf.fit(reg=r)
            try:
                print("Evaluate train:")
                train_pred = clf.predict(train_sentences)
                train_acc = train_pred['accuracy']
                print("Evaluate validation:")
                test_pred = clf.predict(validation_sentences)
                test_acc = test_pred['accuracy']
                results[('reg', r, str(occur_dict))] = {'train_acc': train_acc, 'validation_acc': test_acc}
                if test_acc > best_acc:
                    best_acc = test_acc
                    save_load_init_model(clf=clf, filename=best_model)
            except:
                pass
            print("Current results", results)
            print("\n\n")

    print("Final results")
    print(results)


if __name__ == '__main__':
    mode = None
    try:
        mode = sys.argv[1]
    except Exception as e:
        pass

    if mode == '-t':
        training()
    elif mode == '-t2':
        training2()
    else:
        main()
