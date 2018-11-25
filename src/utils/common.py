def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()


def create_all_features(feature_functions, sentence_tuples, sentence):
    X = []

    for feature_function in feature_functions:
        X.append(feature_function(sentence_tuples))

    print(X)


