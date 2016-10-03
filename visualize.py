"""Visualize emoji clusters using TSNE"""

# External dependencies
import sklearn.manifold as man
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle as pk

# Internal dependencies
from model import Emoji2Vec
from parameter_parser import CliParser

# Authorship
__author__ = "Ben Eisner, Tim Rocktaschel"
__email__ = "beisner@princeton.edu"


def __visualize():
    # setup
    args = CliParser()
    args.print_params('EMOJI VISUALIZATION')

    # mapping from emoji index to emoji
    mapping = pk.load(open(args.mapping_file, 'rb'))

    model = Emoji2Vec(args.model_params, len(mapping), embeddings_array=None, use_embeddings=False)

    # load emoji vectors from the session
    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, args.model_params.model_folder(args.dataset) + '/model.ckpt')

        V = session.run(model.V)

    # plot the emoji using TSNE
    fig = plt.figure()
    ax = fig.add_subplot(111)
    tsne = man.TSNE(perplexity=50, n_components=2, init='random', n_iter=300000, early_exaggeration=1.0,
                    n_iter_without_progress=1000)
    trans = tsne.fit_transform(V)
    x, y = zip(*trans)
    plt.scatter(x, y, marker='o', alpha=0.0)

    for i in range(len(trans)):
        ax.annotate(mapping[i], xy=trans[i], textcoords='data')

    plt.grid()
    plt.show()

if __name__ == '__main__':
    __visualize()
