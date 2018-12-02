import os
from os.path import join, dirname

from src.utils.decoder import CompData
from src.utils.classifier import MaximumEntropyClassifier

if os.name == 'nt':
    path = join(dirname(os.getcwd()), 'resources', 'train_dev.wtag')
else:
    path = join(os.getcwd(), 'resources', 'train_dev.wtag')

iterable_sentences = CompData(path)
clf = MaximumEntropyClassifier()
clf.fit(iterable_sentences)
