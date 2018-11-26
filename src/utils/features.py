def unigram_f(**kwargs):
    """
    This function returns for each word one hot encoding for the POS its labeled with
    :param kwargs: in 'tuples' we get the labeled three for each words
    :return: One hot encoding for a word (on a tag)
    """
    lst = []
    poss = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
            'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
            'WDT', 'WP', 'WP$', 'WRB']
    sentence_tuples = kwargs['tuples']

    for pos in poss:
        word_pos_lst = [0 for i in range(len(sentence_tuples))]
        for idx, word in enumerate(sentence_tuples):
            if word[2] == pos:
                word_pos_lst[idx] = 1

        lst.append(word_pos_lst)

    return lst


def bigram_f(**kwargs):
    sentence_tuples = kwargs['tuples']
    sentence = kwargs['text']
    lst = []

    for idx, sentence_tuple in enumerate(sentence_tuples):
        if idx != 0:
            if sentence[idx - 1] == 'the' and sentence_tuple[2] == 'NN':
                lst.append(1)
            else:
                lst.append(0)
        else:
            lst.append(0)

    return lst
