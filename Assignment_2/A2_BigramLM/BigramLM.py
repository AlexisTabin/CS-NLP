import numpy as np
import unicodedata


class Dictionary(object):
    UNK = '<UNK>'
    BOS = '<BOS>'
    EOS = '<EOS>'

    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)

    def __init__(self):
        self.tok2ind = {self.UNK: 0, self.BOS: 1, self.EOS: 2}
        self.ind2tok = {0: self.UNK, 1: self.BOS, 2: self.EOS}

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return self.normalize(key) in self.tok2ind

    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key, self.UNK)
        if type(key) == str:
            return self.tok2ind.get(self.normalize(key),
                                    self.tok2ind.get(self.UNK))

    def __setitem__(self, key, item):
        if type(key) == int and type(item) == str:
            self.ind2tok[key] = item
        elif type(key) == str and type(item) == int:
            self.tok2ind[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add(self, token):
        token = self.normalize(token)
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def tokens(self):
        """Get dictionary tokens.
        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        tokens = [k for k in self.tok2ind.keys()
                  if k not in {'<NULL>', '<UNK>'}]
        return tokens


class BigramLM(object):
    def __init__(self):
        self.dict = Dictionary()
        self.V = 15087
        self.LM = np.ones((self.V, self.V))

    def train(self, sents):
        for sent in sents:
            for wc, word in enumerate(sent):
                if wc == 0:
                    self.dict.add(word)
                    word_id = self.dict[word]
                    self.LM[1][word_id] += 1
                else:
                    self.dict.add(word)
                    word_id = self.dict[word]
                    prev_word_id = self.dict[sent[wc - 1]]
                    self.LM[prev_word_id][word_id] += 1
                if wc == len(sent) - 1:
                    self.dict.add(word)
                    word_id = self.dict[word]
                    self.LM[word_id][2] += 1
        for i in range(self.V):
            self.LM[i][1] = 0  # bos never appears after a word
            self.LM[2][i] = 0  # eos never appears before anything
        for i in range(self.V):
            denom = np.sum(self.LM[i, :])
            if denom > 0:
                self.LM[i][:] /= denom

    def generate(self, prev_word):
        return self.LM[self.dict[prev_word]][:]


