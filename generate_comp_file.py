from postagger.utils.classifier import save_load_init_model
from postagger.utils.common import get_data_path, timeit, generate_comp_files
from postagger.utils.decoder import CompData


@timeit
def main():
    comp_file = ['comp.words', 'comp2.words']
    model_name = ['best_model.pickle', 'best_model2.pickle']
    model_output = ['comp.wtag', 'comp2.wtag']
    for model_idx, model_name in enumerate(model_name):
        clf = save_load_init_model(clf=None, filename=model_name)

        test_path = get_data_path(comp_file[model_idx])
        test_sentences = CompData(test_path, comp=True)
        t_predict = clf.predict(test_sentences, comp=True)

        generate_comp_files(test_sentences, t_predict['prediction'], model_output[model_idx])


if __name__ == '__main__':
    main()
