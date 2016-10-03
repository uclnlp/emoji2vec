#!/usr/bin/env python
"""Wrapper for word2vec and emoji2vec models, so that we can query by entire phrase, rather than by
individual words.
"""

# External dependencies
import os.path
import gensim.models as gs
import numpy as np

# Authorship
__author__ = "Ben Eisner, Tim Rocktaschel"
__email__ = "beisner@princeton.edu"


class Phrase2Vec:
    """Wrapper for the word2vec model and emoji2vec model, allowing us to compute phrases"""
    def __init__(self, dim, w2v, e2v=None):
        """Constructor for the Phrase2Vec model

        Args:
            dim: Dimension of the vectors in word2vec and emoji2vec
            w2v: Gensim object for word2vec
            e2v: Gensim object for emoji2vec
        """
        self.wordVecModel = w2v
        if e2v is not None:
            self.emojiVecModel = e2v
        else:
            self.emojiVecModel = dict()
        self.dimension = dim

    @classmethod
    def from_word2vec_paths(cls, dim, w2v_path='/data/word2vec/GoogleNews-vectors-negative300.bin',
                            e2v_path=None):
        """Creates a Phrase2Vec object based on paths for w2v and e2v

        Args:
            dim: Dimension of the vectors in word2vec and emoji2vec
            w2v_path: Path to word2vec vectors
            e2v_path: Path to emoji2vec vectors

        Returns:

        """
        if not os.path.exists(w2v_path):
            print(str.format('{} not found. Either provide a different path, or download binary from '
                             'https://code.google.com/archive/p/word2vec/ and unzip', w2v_path))

        w2v = gs.Word2Vec.load_word2vec_format(w2v_path, binary=True)
        if e2v_path is not None:
            e2v = gs.Word2Vec.load_word2vec_format(e2v_path, binary=True)
        else:
            e2v = dict()
        return cls(dim, w2v, e2v)

    def __getitem__(self, item):
        """Get the vector sum of all tokens in a phrase

        Args:
            item: Phrase to be converted into a vector sum

        Returns:
            phr_sum: Bag-of-words sum of the tokens in the phrase supplied
        """
        tokens = item.split(' ')
        phr_sum = np.zeros(self.dimension, np.float32)

        for token in tokens:
            if token in self.wordVecModel:
                phr_sum += self.wordVecModel[token]
            elif token in self.emojiVecModel:
                phr_sum += self.emojiVecModel[token]

        return phr_sum

    def from_emoji(self, emoji_vec, top_n=10):
        """Get the top n closest tokens for a supplied emoji vector

        Args:
            emoji_vec: Emoji vector
            top_n: number of results to return

        Returns:
            Closest n tokens for a supplied emoji_vec
        """
        return self.wordVecModel.most_similar(positive=emoji_vec, negative=[], topn=top_n)

    def __setitem__(self, key, value):
        self.wordVecModel[key] = value
