import os
from src.preprocessing import clean_doc
import collections

class IMDB:
    
    def __init__(self):
        negative_docs = self._read_files('data/imdb_50k/train/neg')
        positive_docs = self._read_files('data/imdb_50k/train/pos')
        negative_docs_test = self._read_files('data/imdb_50k/test/neg')
        positive_docs_test = self._read_files('data/imdb_50k/test/pos')
        
        self.data = negative_docs + positive_docs
        self.labels = [0 for _ in range(len(negative_docs))] + [1 for _ in range(len(positive_docs))]

        self.test_data = negative_docs_test + positive_docs_test
        self.test_labels = [0 for _ in range(len(negative_docs_test))] + [1 for _ in range(len(positive_docs_test))]
        
        self.vocab = self._get_vocab(self.data)
    
    def _read_files(self, path):
        documents = list()
        if os.path.isdir(path):
            for filename in os.listdir(path):
                with open('%s/%s' % (path, filename), encoding='iso-8859-1') as f:
                    doc = f.read()
                    doc = clean_doc(doc)
                    documents.append(doc)
        
        if os.path.isfile(path):        
            with open(path, encoding='iso-8859-1') as f:
                doc = f.readlines()
                for line in doc:
                    documents.append(clean_doc(line))
        return documents

    def _get_vocab(self, dataset):
        vocab = {}

        for example in dataset:
            example_as_list = example.split()
            for word in example_as_list:
                vocab[word] = 0

        for example in dataset:
            example_as_list = example.split()
            for word in example_as_list:
                vocab[word] += 1

        return collections.OrderedDict(sorted(vocab.items(), key=lambda x: x[1], reverse=True))

    def getData(self):
        return self.data + self.test_data, self.labels + self.test_labels