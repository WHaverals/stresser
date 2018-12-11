import collections
import json

import numpy as np

SYMBOLS = PAD, BOS, EOS, UNK = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']

class SequenceVectorizer:
    def __init__(self, min_cnt = 0,
                 max_len = None, syll2idx = None,
                 idx2syll =  None):
        self.min_cnt = min_cnt
        self.max_len = max_len
        self.syll2idx = syll2idx if syll2idx is not None else {}
        self.idx2syll = idx2syll if idx2syll is not None else {}
        self.fitted = len(self.syll2idx) > 0

    def fit(self, lines):
        if self.fitted:
            return self

        counter = collections.Counter()
        for line in lines:
            counter.update(line)
        
        if not self.max_len:
            self.max_len = len(max(lines, key=len)) + 2

        self.syll2idx = {}
        for syll in SYMBOLS + sorted(k for k, v in counter.most_common()
                                     if v >= self.min_cnt):
            self.syll2idx[syll] = len(self.syll2idx)

        # construct a dict mapping indices to characters:
        self.idx2syll = {i: s for s, i in self.syll2idx.items()}
        return self

    def transform(self, lines):
        X = []
        for line in lines:
            x = [self.syll2idx['<BOS>']]
            for syll in line:
                x.append(self.syll2idx.get(syll, self.syll2idx['<BOS>']))
                # truncate longer tokens
                if len(x) >= (self.max_len - 1):
                    break
            x.append(self.syll2idx['<EOS>'])
            # right-pad shorter sequences
            X.append(x + [self.syll2idx[PAD]] * (self.max_len - len(x)))

        return np.array(X, dtype=np.int32)
    
    def normalize_len(self, labels):
        X, x = [], []
        for line in labels:
            x = [0] + line[:self.max_len - 2] + [0]
            while len(x) < self.max_len:
                x.append(0)
            X.append(x)
        return np.array(X, dtype=np.float32)

    def inverse_transform(self, lines):
        return [[self.idx2syll[int(idx)] for idx in line] for line in lines]

    def fit_transform(self, tokens):
        return self.fit(tokens).transform(tokens)

    def dump(self, path):
        with open(path, 'w') as f:
            json.dump(
                {
                    'min_cnt': self.min_cnt,
                    'max_len': self.max_len,
                    'idx2syll': self.idx2syll,
                    'syll2idx': self.syll2idx
                },
                f, indent=4)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            params = json.load(f)
        return cls(**params)
