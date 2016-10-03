#!/usr/bin/env python
"""Argument parser for all parameters one could pass to functions in this package"""

# External dependencies
import argparse as arg
import pprint

# Internal dependencies
from model import ModelParams

# Authorship
__author__ = "Ben Eisner, Tim Rocktaschel"
__email__ = "beisner@princeton.edu"


class CliParser:
    """Parameter parser for all arguments"""
    def __init__(self):
        self.parser = arg.ArgumentParser(description='Parser for training/evaluationg emoji2vec model')

        # Directories/files
        self.parser.add_argument('-d', '--dir', default='./data/training/', type=str,
                                 help='directory for training data')
        self.parser.add_argument('-w', '--word', default='./data/word2vec/GoogleNews-vectors-negative300.bin',
                                 type=str, help='path to the word2vec file')
        self.parser.add_argument('-m', '--mapping', default='emoji_mapping.p', type=str,
                                 help='emoji index mapping file')
        self.parser.add_argument('-em', '--embeddings', default='generated_embeddings.p', type=str,
                                 help='file for generated embeddings')

        # Model parameters
        self.parser.add_argument('-k', '--dim', default=300, type=int, help='train a 300 x k projection matrix')
        self.parser.add_argument('-b', '--batch', default=4, type=int,
                                 help='positive examples in minibatch (total size=batch*(1+ratio)')
        self.parser.add_argument('-e', '--epochs', default=40, type=int, help='number of training epochs')
        self.parser.add_argument('-r', '--ratio', default=1, type=int, help='ratio of negative examples to positive')
        self.parser.add_argument('-l', '--learning', default=0.001, type=float, help='learning rate')
        self.parser.add_argument('-dr', '--dropout', default=0.1, type=float, help='amount of dropout to use')
        self.parser.add_argument('-t', '--threshold', default=0.5, type=float,
                                 help='threshold for binary classification')

        # Miscellaneous
        self.parser.add_argument('-ds', '--dataset', default='unicode', type=str, help='unicode or emojipedia')
        self.parser.add_argument('-D', '--debug', help='enable debugging')

        args = self.parser.parse_args()

        # dimensions of projected embeddings
        self.model_params = ModelParams(300, out_dim=args.dim, pos_ex=args.batch, max_epochs=args.epochs,
                                        neg_ratio=args.ratio, learning_rate=args.learning, dropout=args.dropout,
                                        class_threshold=args.threshold)

        # debug mode?
        self.debug = args.debug

        # data folder
        self.data_folder = args.dir

        # file for generated embeddings
        self.embeddings_file = args.embeddings

        # file for emoji mappings
        self.mapping_file = args.mapping

        # word2vec file
        self.word2vec_file = args.word

        # dataset to chose: unicode or emojipedia
        self.dataset = args.dataset

    def print_params(self, title):
        """Pretty-print the parameters

        Args:
            title: Pretty-print header
        """
        print('-' * 80)
        print(title)
        print('-' * 80)
        print('Hyperparameters:')
        print('k\t', self.model_params.out_dim)
        print('pos_ex\t', self.model_params.pos_ex)
        print('ratio\t', self.model_params.neg_ratio)
        print('mb\t', self.model_params.mb)
        print('epochs\t', self.model_params.max_epochs)
        print('learning\t', self.model_params.learning_rate)
        print('dropout\t', self.model_params.dropout)
        print('dataset\t', self.dataset)

    def print_search_params(self, title, search_params):
        """Pretty-print the search parameters

        Args:
            title: Pretty-print header
            search_params: search parameters
        """
        print('-' * 80)
        print(title)
        print('-' * 80)
        print('Hyperparameters:')
        print('learning\t', self.model_params.learning_rate)
        pprint.pprint(search_params)
