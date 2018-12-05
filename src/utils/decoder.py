from .common import read_file


class Dataset(object):
    """Class that iterates over Dataset
    __iter__ method yields a tuple (words, tags)
        lst: list of words/tags
        sentence: the text of the sentence
    If processing_word and processing_tag are not None,
    optional preprocessing is applied
    Example:
        ```python
        data = Dataset(filename)
        for tuples, sentence in data:
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


class CompData(Dataset):
    def get_comp_text(self):
        return read_file(self.filename)

    def __iter__(self):
        """
        iterates over text files, for each position i in sentence create a history tuple (X)
        and a label (y)
        :return: X, y, sentences
        """
        txt = self.get_comp_text()
        # split by sentences
        sentences = txt.split('\n')

        for sent_id, sentence in enumerate(sentences):
            words = sentence.split(' ')
            stripped_sentence = []
            # X, y
            tuples, tags = [], []
            # helper
            word_tag_tuples = []

            for word in words:
                word_stripped, tag_stripped = word.split('_')
                word_tag_tuples.append((word_stripped, tag_stripped))
                stripped_sentence.append(word_stripped)

            for i, word_tag_tuple in enumerate(word_tag_tuples):
                # word = word_tag_tuple[0]
                tag = word_tag_tuple[1]
                tags.append(tag)
                if i == 0:
                    tuples.append(('*', '*', sent_id, i))
                elif i == 1:
                    tuples.append(('*', word_tag_tuples[i - 1][1], sent_id, i))
                else:
                    u = word_tag_tuples[i - 2][1]  # pre pre tag
                    v = word_tag_tuples[i - 1][1]  # pre tag
                    tuples.append((u, v, sent_id, i))

            yield tuples, tags, stripped_sentence

    def get_tags(self):
        tags_tmp = []

        for tuples, tags, sentence in self:
            for i in range(len(tuples)):
                tags_tmp.append(tags[i])

        return tags_tmp

    def get_sentences(self):
        sentences_tmp = []

        for tuples, tags, sentence in self:
            sentences_tmp.append(sentence)

        return sentences_tmp
