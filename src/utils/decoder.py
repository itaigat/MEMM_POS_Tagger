from .common import read_file


class Dataset(object):
    """Class that iterates over Dataset
    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags
    If processing_word and processing_tag are not None,
    optional preprocessing is applied
    Example:
        ```python
        data = Dataset(filename)
        for sentence, tags in data:
            pass
        ```
    """

    def __init__(self, filename, max_iter=None):
        """
        Args:
            filename: path to the file
            max_iter: (optional) max number of sentence to yield
        """
        self.filename = filename
        self.max_iter = max_iter
        self.length = None

    def __iter__(self):
        pass

    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length

    def __str__(self):
        st = ''
        for sentence, tag, pos in self:
            st += ' '.join(sentence) + '\n' + ' '.join(tag) + '\n' + ' '.join(pos) + '\n'

        return st


class Comp(Dataset):
    def __iter__(self):
        txt = read_file(self.filename)
        sentences = txt.split('\n')
        for sentence in sentences:
            tuples, labeled_words, sentence_lst = [], [], []
            words = sentence.split(' ')

            for word in words:
                labeled_words.append(word.split('_'))

            for idx, labeled_word in enumerate(labeled_words):
                if idx == 0:
                    tuples.append(('*', '*', labeled_word[1]))
                elif idx == 1:
                    tuples.append(('*', labeled_words[0][1], labeled_word[1]))
                else:
                    tuples.append((labeled_words[idx - 2][1], labeled_words[idx - 1][1], labeled_word[1]))

                sentence_lst.append(labeled_word[0])

            yield tuples, sentence_lst
