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
model_name = 'model_dev.pickle'
model_matrices = 'model_matrices_dev.pickle'
model_preprocess = 'model_preprocess_dev.pickle'
verbose = 1

# data files
train = 'train2.wtag'
test = 'test.wtag'
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
    train_sentences = CompData(train_path, slice=(0, 630))

    validation_sentences = CompData(train_path, slice=(630, 700))

    preprocessor = load_save_preprocessed_data(model_preprocess, train_sentences, load=load_preprocess)
    # apply filtering
    pdict = preprocessor.summarize_counts(method='cut', dict=min_occurrence_dict)
    # init classifier with known tags
    tags = get_tags(train)
    clf = MaximumEntropyClassifier(train_sentences, pdict, tags)

    reg = [1e-2, 1e-1, 1, 3, 5, 10, 25, 50, 100, 500, 1000]
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
            print("Evaluate validation:")
            test_pred = clf.predict(validation_sentences)
            test_acc = test_pred['accuracy']
            results[('reg', r)] = {'train_acc': train_acc, 'validation_acc': test_acc}
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
