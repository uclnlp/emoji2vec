#!/usr/bin/env python3
"""Library for performing a grid search on a parameter space with the model

Specify the grid search parameters when calling grid_search. The format for search params is:

search_params = {
    "out_dim": [300],
    "pos_ex": [4, 16, 64],
    "max_epochs": [10, 20],
    "ratio": [0, 1, 2],
    "dropout": [0.0, 0.1]
}

"""

# External dependencies
import tensorflow as tf
import pickle as pk
import math

# Internal dependencies
import parameter_parser as pp
from model import ModelParams

# Authorship
from train import train_save_evaluate
from utils import get_examples_from_kb, build_kb, generate_embeddings

__author__ = "Ben Eisner, Tim Rocktaschel"
__email__ = "beisner@princeton.edu"

# The grid search we will perform will iterate through all permutations of these parameters
search_params = {
    "out_dim": [300],
    "pos_ex": [4, 16, 64],
    "max_epochs": [10, 20],
    "ratio": [0, 1, 2],
    "dropout": [0.0, 0.1]
}


# Recursive Grid Search
def __grid_search(remaining_params, current_params, results_dict, train_set, dev_set, kb, embeddings_array, ind2emoji,
                  dataset_name, in_dim, learning_rate, threshold):
    if len(remaining_params) > 0:
        # Get a parameter
        param, values = remaining_params.popitem()

        # For each potential parameter, copy current_params and add the potential parameter to next_params
        for value in values:
            next_params = current_params.copy()
            next_params[param] = value

            # Perform grid search on the remaining params
            __grid_search(remaining_params=remaining_params.copy(), current_params=next_params,
                          results_dict=results_dict, train_set=train_set, dev_set=dev_set, kb=kb,
                          embeddings_array=embeddings_array, ind2emoji=ind2emoji, dataset_name=dataset_name,
                          in_dim=in_dim, learning_rate=learning_rate, threshold=threshold)
    else:
        model_params = ModelParams(in_dim=in_dim, out_dim=current_params["out_dim"],
                                   max_epochs=current_params["max_epochs"], pos_ex=current_params["pos_ex"],
                                   neg_ratio=current_params["ratio"], learning_rate=learning_rate,
                                   dropout=current_params["dropout"], class_threshold=threshold)

        name = model_params.model_folder(dataset_name)
        # We know that the larger the batch size, the more epochs needed to convergence, therefore we modify the batch
        # size here
        model_params.max_epochs = int(model_params.max_epochs * math.sqrt(model_params.pos_ex)
                                      * (model_params.neg_ratio + 1))

        results_dict[name] = train_save_evaluate(params=model_params, train_set=train_set, dev_set=dev_set,
                                                 kb=kb, embeddings_array=embeddings_array, ind2emoji=ind2emoji,
                                                 dataset_name=dataset_name)

    return results_dict


def grid_search(params, learning_rate, threshold, in_dim, kb, embeddings_array, ind2emoji, dataset_name):
    """Perform a grid search on the search parameter space provided py params

    Args:
        params: Dictionary of parameters to potential values
        learning_rate: Learning rate
        threshold: Threshold for accuracy
        in_dim: Dimension of word2vec model
        kb: Knowledge Base
        embeddings_array: Embeddings
        ind2emoji: Mapping from index to emoji
        dataset_name: Name of the dataset

    Returns:
        A dictionary of the models to their metrics
    """
    # Set the TensorFlow graph as default
    tf.Graph().as_default()

    # Mapping between parameter set and metrics.
    results_dict = dict()

    # Get examples of each example type in two sets. This is just a reprocessing of the knowledge base for efficiency,
    # so we don't have to generate the train and dev set on each train
    train_set = get_examples_from_kb(kb=kb, example_type='train')
    dev_set = get_examples_from_kb(kb=kb, example_type='dev')

    # Perform the recursive grid search
    return __grid_search(remaining_params=params, current_params=dict(), results_dict=results_dict,
                         train_set=train_set, dev_set=dev_set, kb=kb, embeddings_array=embeddings_array,
                         ind2emoji=ind2emoji, dataset_name=dataset_name, in_dim=in_dim, learning_rate=learning_rate,
                         threshold=threshold)


# Run grid search, only for standalone execution
def __run_grid_search():
    # Read in arguments
    args = pp.CliParser()
    args.print_search_params('EMOJI2VEC GRID SEARCH', search_params)

    # Read in training data, generate mappings, and generate embeddings
    print('reading training data from: ' + args.data_folder)
    kb, ind2phr, ind2emoji = build_kb(args.data_folder)
    pk.dump(ind2emoji, open(args.mapping_file, 'wb'))
    embeddings_array = generate_embeddings(ind2phr=ind2phr, kb=kb, embeddings_file=args.embeddings_file,
                                                 word2vec_file=args.word2vec_file)

    # Perform grid search
    print('performing grid search')
    results_dict = grid_search(params=search_params, learning_rate=args.model_params.learning_rate,
                               threshold=args.model_params.class_threshold, in_dim=args.model_params.in_dim, kb=kb,
                               embeddings_array=embeddings_array, ind2emoji=ind2emoji, dataset_name=args.dataset)

    # Get top 5 results
    results = sorted(results_dict, key=(lambda x: results_dict[x]['auc']), reverse=True)
    for result in results[:5]:
        print(str.format('{}\n{}', result, results_dict[result]))

    m = results_dict[results[0]]
    print(str.format("The best combination, by auc score, is: {} at {}", results[0], m))


if __name__ == '__main__':
    __run_grid_search()
