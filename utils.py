"""Utility functions for training and evaluation"""

# External dependencies
import pickle as pk
from sklearn import metrics

import numpy as np
import os.path
from gensim import matutils

from naga.shared.kb import KB

# Internal dependencies
from phrase2vec import Phrase2Vec

# Authorship
__author__ = "Ben Eisner, Tim Rocktaschel"
__email__ = "beisner@princeton.edu"


def generate_embeddings(ind2phr, kb, embeddings_file, word2vec_file, word2vec_dim=300):
    """Generate a numpy array of phrase embeddings for all phrases in the knowledge base.

        Since it is expensive to calculate these phrase embeddings every time, we cache the output
        in a file, which we can load from if this function is called on a set we've seen before.

    Args:
        word2vec_dim: Dimension of the static word2vec model we use
        ind2phr: Mapping from phrase indices to phrases in the KB
        kb: Knowledge base
        embeddings_file: File where we store the embeddings
        word2vec_file: word2vec model file

    Returns:

    """
    phrase_vector_sums = dict()

    # get the complete word vectors from the second argument
    if not (os.path.isfile(embeddings_file)):
        print('reading embedding data from: ' + word2vec_file)
        phrase_vec_model = Phrase2Vec.from_word2vec_paths(word2vec_dim, w2v_path=word2vec_file)

        print('generating vector subset')
        for phrase in kb.get_vocab(1):
            phrase_vector_sums[phrase] = phrase_vec_model[phrase]

        pk.dump(phrase_vector_sums, open(embeddings_file, 'wb'))
    else:
        print('loading embeddings...')
        phrase_vector_sums = pk.load(open(embeddings_file, 'rb'))

    # build the embeddings array, for lookup later
    embeddings_array = np.zeros(shape=[len(ind2phr), 300], dtype=np.float32)
    for ind, phr in ind2phr.items():
        embeddings_array[ind] = phrase_vector_sums[phr]

    return embeddings_array


# Read data from a file and inject it into a knowledge base
def __read_data(filename, base, ind_to_phr, ind_to_emoj, typ):
    with open(filename, 'r') as f:
        # build the data line by line
        lines = f.readlines()
        for line in lines:
            ph, em, truth = line.rstrip().split('\t')
            base.add((truth == 'True'), typ, em, ph)
            ind_to_phr[base.get_id(ph, 1)] = ph
            ind_to_emoj[base.get_id(em, 0)] = em


def build_kb(data_folder):
    """Read training data from the training directory and generate a KB

    Args:
        data_folder: Directory containing a train.txt, a dev.txt, and a test.txt from which
            we can assemble our knowledge base.
    """
    base = KB()

    # KB indices to phrase
    ind_to_phr = dict()

    # KB indices to emoji
    ind_to_emoj = dict()

    __read_data(data_folder + 'train.txt', base, ind_to_phr, ind_to_emoj, 'train')
    __read_data(data_folder + 'dev.txt', base, ind_to_phr, ind_to_emoj, 'dev')
    __read_data(data_folder + 'test.txt', base, ind_to_phr, ind_to_emoj, 'test')

    return base, ind_to_phr, ind_to_emoj


def get_examples_from_kb(kb, example_type='train'):
    """Extract all the examples of a type (i.e. train, dev, test) from the knowledge base

    Args:
        kb: Knowledge base
        example_type: Name of example type (i.e. train, dev, test)

    Returns:
        Lists of the rows, columns, and targets from the dataset
    """
    # prepare the training set
    batch = list(kb.get_all_facts([example_type]))
    rows = list()
    cols = list()
    targets = list()

    for i in range(len(batch)):
        example = batch[i]
        cols.append(kb.get_id(example[0][0], 0))
        rows.append(kb.get_id(example[0][1], 1))
        targets.append(example[1])

    return rows, cols, targets


def __sigmoid(x):
    return 1 / (1 + np.math.exp(-x))


def generate_predictions(e2v, dset, phr_embeddings, ind2emoji, threshold):
    """Calculate whether a set of emoji/phrase pairs are correlated

        This implementation doesn't use TensorFlow, and relies instead of injected vectors

    Args:
        e2v: Mapping from emoji to vector, typically the trained emoji vectors from our model.
        dset: KB that contains pairs of emoji and phrases, as well as whether they are correlated.
        phr_embeddings: Map between phrase indices and phrase vectors, as computed by the vector sum of
            word vectors for that phrase.
        ind2emoji: Map between emoji index and emoji, for converting dset indices into emoji.
        threshold: Threshold for classifying correlation as true or false.

    Returns:
        y_pred_labels: List of predicted labels for pairs in the dataset
        y_pred_values: List of predicted scores for pairs in the dataset
        y_true_labels: List of true labels for pairs in the dataset
        y_true_values: List of true scores for pairs in the dataset

    """
    y_pred_labels = list()
    y_pred_values = list()

    phr_ixs, em_ixs, truths = dset

    for (phr_ix, em_ix, truth) in zip(phr_ixs, em_ixs, truths):
        prob = __sigmoid(
            np.dot(matutils.unitvec(phr_embeddings[phr_ix]), matutils.unitvec(e2v[ind2emoji[em_ix]])))
        y_pred_values.append(prob)
        y_pred_labels.append(prob >= threshold)  # Threshold predicted probability

    y_true_values = [float(v) for v in truths]

    return y_pred_labels, y_pred_values, truths, y_true_values


def get_metrics(pred_labels, pred_values, truth_labels, truth_values):
    """Get a set of metrics, including accuracy, f1 score, and area under the curve.

    This method takes in predictions and spits out performance metrics.

    Args:
        pred_labels: Predicted labels for correlation between an emoji and a phrase.
        pred_values: Predicted correlation value between an emoji and a phrase.
        truth_labels: True labels for correlation between an emoji and a phrase.
        truth_values: True correlation value between an emoji and a phrase

    Returns:

    """
    acc = metrics.accuracy_score(y_true=truth_labels, y_pred=pred_labels)
    f1 = metrics.f1_score(y_true=truth_labels, y_pred=pred_labels)
    try:
        auc = metrics.roc_auc_score(y_true=truth_values, y_score=pred_values)
    except:
        auc = 'N/A'

    return acc, f1, auc
