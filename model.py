#!/usr/bin/env python
"""emoji2vec model implemented in TensorFlow.

File also contains a ModelParams class, which is a convenience wrapper for all the parameters to the model.
Details of the model can be found below.
"""

# External dependencies
import tensorflow as tf
import numpy as np
import sklearn.metrics as metrics
import gensim.models as gs

from naga.shared.kb import BatchNegSampler
from naga.shared.trainer import Trainer

# Authorship
__author__ = "Ben Eisner, Tim Rocktaschel"
__email__ = "beisner@princeton.edu"


class ModelParams:
    """Convenience class for passing around model parameters"""

    def __init__(self, in_dim, out_dim, max_epochs, pos_ex, neg_ratio, learning_rate, dropout, class_threshold):
        """Create a struct of all parameters that get fed into the model

        Args:
            in_dim: Dimension of the word vectors supplied to the algorithm (i.e. word2vec)
            out_dim: Dimension of the output emoji vectors of the algorithm
            pos_ex: Number of positive examples per batch
            max_epochs: Max number of training epochs
            neg_ratio: Ratio between negative examples and positive examples in a batch
            learning_rate: Learning rate
            dropout: Dropout rate
            class_threshold: Classification threshold for accuracy
        """
        self.class_threshold = class_threshold
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.neg_ratio = neg_ratio
        self.max_epochs = max_epochs
        self.pos_ex = pos_ex
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.mb = self.pos_ex * (1 + self.neg_ratio)

    def model_folder(self, dataset_name):
        """Get the model path for a given dataset

        Args:
            dataset_name: The name of the dataset we used to generate training data

        Returns:
            The model path for a given dataset
        """
        return str.format(
            str.format('./results/{}/k-{}_pos-{}_rat-{}_ep-{}_dr-{}', dataset_name, self.out_dim, self.pos_ex,
                       self.neg_ratio, self.max_epochs, int(self.dropout * 10)))


class Emoji2Vec:
    """Class for representing the model in TensorFlow."""
    # TODO(beisner): Describe the model in more detail here

    # define the model
    def __init__(self, model_params, num_emoji, embeddings_array, use_embeddings=True):
        """Constructor for the Emoji2Vec model

        Args:
            model_params: Parameters for the model
            num_emoji: Number of emoji we will ultimately train
            embeddings_array: For quick training, we inject a constant array into TensorFlow graph consisting
                of vector sums of the embeddings
            use_embeddings: If True, embeddings must be passed in, but the model will not accept arbitrary queries
                If false, it will accept arbitrary queries
        """

        self.model_params = model_params
        self.num_cols = num_emoji
        self.embeddings_array = embeddings_array

        # If we are trying to learn the emoji in the same space as the words, we don't need a projection matrix
        # also saves some training time
        is_proj = not (model_params.in_dim == model_params.out_dim)

        # Phrase indices in current batch
        self.row = tf.placeholder(tf.int32, shape=[None], name='row')

        # Phrase vectors in current batch (optional input to the graph)
        self.orig_vec = tf.placeholder(tf.float32, shape=[None, model_params.in_dim], name='orig_vec')

        # Emoji indices in current batch
        self.col = tf.placeholder(tf.int32, shape=[None], name='col')

        # Correlation between an emoji and a phrase
        self.y = tf.placeholder(tf.float32, shape=[None], name='y')

        # Column embeddings (here emoji representations)
        self.V = tf.Variable(tf.random_uniform([num_emoji, model_params.out_dim], -0.1, 0.1), name="V")

        # original phrase embeddings from Word2Vec, dependent on parameter
        if use_embeddings:
            # constant embeddings
            tf_embeddings = tf.constant(embeddings_array)
            orig_vec = tf.nn.embedding_lookup(tf_embeddings, self.row)
        else:
            orig_vec = self.orig_vec

        if is_proj:
            # Row embeddings (here phrase representations)
            # Don't need to learn this if we stay in the same space
            W = tf.Variable(tf.random_uniform([model_params.in_dim, model_params.out_dim], -0.1, 0.1), name="W")
            v_row = tf.tanh(tf.matmul(orig_vec, W))
        else:
            v_row = orig_vec

        v_col = tf.nn.embedding_lookup(self.V, self.col)
        v_col = tf.nn.dropout(v_col, (1 - model_params.dropout))

        # Calculate the predicted score, a.k.a. dot product (here)
        self.score = tf.reduce_sum(tf.mul(v_row, v_col), 1)

        # Probability of match
        self.prob = tf.sigmoid(self.score)

        # Calculate the cross-entropy loss
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(self.score, self.y)

    # train the model using the appropriate parameters
    def train(self, kb, hooks, session):
        """Train the model on a given knowledge base

        Args:
            kb: Knowledge Base, will only train on training examples
            hooks: Hooks to print out metrics at various intervals
            session: TensorFlow session

        """
        optimizer = tf.train.AdamOptimizer(self.model_params.learning_rate)

        # corpus is the body from which we sample
        corpus = BatchNegSampler(kb, 1, self.model_params.mb, self.model_params.neg_ratio)

        trainer = Trainer(optimizer, self.model_params.max_epochs, hooks)
        trainer(corpus, placeholders=[self.col, self.row, self.y], loss=self.loss, model=self.score,
                session=session)

    def predict(self, session, dset, threshold):
        """Generate predictions on a given set of examples using TensorFlow

        Args:
            session: TensorFlow session
            dset: Dataset tuple (emoji_ix, phrase_ix, truth)
            threshold: Threshold for classification

        Returns:
            Returns predicted values for an example, as well as the true value
        """
        phr_ix, em_ix, truth = dset

        res = session.run(self.prob, feed_dict={
            self.col: em_ix,
            self.row: phr_ix
        })

        y_pred = [1 if y > threshold else 0 for y in res]
        y_true = np.asarray(truth).astype(int)
        return y_pred, y_true

    def accuracy(self, session, dset, threshold=0.5):
        """Calculate the accuracy of a dataset at a given threshold.

        Args:
            session: TensorFlow session
            dset: Dataset tuple (emoji_ix, phrase_ix, truth)
            threshold: Threshold for classification

        Returns:
            Accuracy
        """
        y_pred, y_true = self.predict(session, dset, threshold)
        return metrics.accuracy_score(y_true=y_true, y_pred=y_pred)

    def f1_score(self, session, dset, threshold=0.5):
        """Calculate the f1 score of a dataset at a given classification threshold.

        Args:
            session: TensorFlow session
            dset: Dataset tuple (emoji_ix, phrase_ix, truth)
            threshold: Threshold for classification

        Returns:
            F1 score
        """
        y_pred, y_true = self.predict(session, dset, threshold)
        return metrics.f1_score(y_true=y_true, y_pred=y_pred)

    def auc(self, session, dset):
        """Calculates the Area under the Curve for the f1 at various thresholds

        Args:
            session: TensorFlow session
            dset: Dataset tuple (emoji_ix, phrase_ix, truth)

        Returns:

        """
        phr_ix, em_ix, truth = dset
        res = session.run(self.prob, feed_dict={
            self.col: em_ix,
            self.row: phr_ix
        })

        y_true = np.asarray(truth).astype(int)
        return metrics.roc_auc_score(y_true, res)

    def roc_vals(self, session, dset):
        """Generates a receiver operating curve for the dataset

        Args:
            session: TensorFlow session
            dset: Dataset tuple (emoji_ix, phrase_ix, truth)

        Returns:
            Points on the curve
        """
        phr_ix, em_ix, truth = dset
        res = session.run(self.prob, feed_dict={
            self.col: em_ix,
            self.row: phr_ix
        })

        y_true = np.asarray(truth).astype(int)
        return metrics.roc_curve(y_true, res)

    def create_gensim_files(self, sess, model_folder, ind2emoj, out_dim):
        """Given a trained session and a destination path (model_folder), generate the gensim binaries
        for a model.

        Args:
            sess: A trained TensorFlow session
            model_folder: Folder in which to generate the files
            ind2emoj: Mapping from indices to emoji
            out_dim: Output dimension of the emoji vectors

        Returns:

        """
        vecs = sess.run(self.V)
        txt_path = model_folder + '/emoji2vec.txt'
        bin_path = model_folder + '/emoji2vec.bin'
        f = open(txt_path, 'w')
        f.write('%d %d\n' % (len(vecs), out_dim))
        for i in range(len(vecs)):
            f.write(ind2emoj[i] + ' ')
            for j in range(out_dim):
                f.write(str.format('{} ', vecs[i][j]))
            f.write('\n')
        f.close()

        e2v = gs.Word2Vec.load_word2vec_format(txt_path, binary=False)
        e2v.save_word2vec_format(bin_path, binary=True)

        return e2v
