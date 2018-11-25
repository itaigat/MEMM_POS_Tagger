def unigram(sentence_tuples):
    lst = []
    for word in sentence_tuples:
        if word[2] == 'NN':
            lst.append(1)
        lst.append(0)
    return lst
