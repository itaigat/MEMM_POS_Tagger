import os
from os.path import join, dirname

from src.utils.decoder import CompData, create_dataset
from src.utils.classifier import MaximumEntropyClassifier

if os.name == 'nt':
    path = join(dirname(os.getcwd()), 'resources', 'train_dev.wtag')
else:
    path = join(os.getcwd(), 'resources', 'train_dev.wtag')

iterable_sentences = CompData(path)
X_train, y_train = create_dataset(iterable_sentences)

# sanity check
print(X_train.shape)
print(y_train.shape)

# at this point we can fit the algorithm
clf = MaximumEntropyClassifier()
clf.fit(X_train, y_train)



