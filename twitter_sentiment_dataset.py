"""Helper class for working with a Twitter dataset"""

# External dependencies
import tweepy
import pickle
import os.path as path
from sklearn import cross_validation
import numpy as np
import nltk.tokenize as tk
import math
import scipy.stats as stats

# Authorship
__author__ = "Ben Eisner, Tim Rocktaschel"
__email__ = "beisner@princeton.edu"


class TweetTrainingExample:
    """Structure holding a Tweet Training example"""

    def __init__(self, id, text, label):
        """Create the training example

        Args:
            id: ID of the example
            text: text of the example
            label: example label
        """
        self.id = id
        self.text = text
        self.label = label

    def __repr__(self):
        return str.format('{}, {}, {}\n', self.id, self.label, self.text)


def get_all_examples():
    """Load all examples from a file at ./data/tweets/examples.p

    Returns:
        A dict of tweets from id to tweet

    """
    save_file = './data/tweets/examples.p'
    if path.exists(save_file):
        ids_to_examples = pickle.load(open(save_file, "rb"))
    else:
        print('Could not find tweets, so returning an empty dict!')
        ids_to_examples = dict()

    return [x for x in list(ids_to_examples.values()) if x.text is not None]


def get_emojis_in_tweet(tweet, emojis_ours, emojis_theirs, emojis_popular, tokenizer):
    """Get a list of all the emojis in a tweet based on the sets provided

    Args:
        tweet: Tweet
        emojis_ours: Emoji vectors trained on our model
        emojis_theirs: Emoji vectors trained on an external model
        emojis_popular: List of popular emojis
        tokenizer: NLTK tokenizer

    Returns:
        Emojis in a tweet
    """
    tokens = tokenizer.tokenize(tweet.text)
    ems = set()
    for token in tokens:
        if token in emojis_ours or token in emojis_theirs or token in emojis_popular:
            ems.add(token)
    return ems


def get_tweets_with_emoji(tweets, emojis_ours, emojis_theirs, emojis_popular):
    """Get all tweets with emoji in the sets

    Args:
        tweets: List of Tweets
        emojis_ours: Emoji vectors trained on our model
        emojis_theirs: Emoji vectors trained on an external model
        emojis_popular: List of popular emojis

    Returns:
        All tweets containing emoji

    """
    tokenizer = tk.TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    ems = list()
    for tweet in tweets:
        if get_emojis_in_tweet(tweet, emojis_ours, emojis_theirs, emojis_popular, tokenizer):
            ems.append(tweet)
    return ems


def num_tweets_with_emoji(tweets, emojis_ours, emojis_theirs, emojis_popular):
    """Get the number of tweets having emojis of the sets provided

    Args:
        tweets: List of Tweets
        emojis_ours: Emoji vectors trained on our model
        emojis_theirs: Emoji vectors trained on an external model
        emojis_popular: List of popular emojis

    Returns:
        Number of tweets with emoji in a dataset.
    """
    ems = get_tweets_with_emoji(tweets, emojis_ours, emojis_theirs, emojis_popular)
    return len(ems)


def save_training_test_sets():
    """Randomly split the dataset into train and test, and save."""
    tweets = get_all_examples()
    train_tweets, test_tweets = cross_validation.train_test_split(tweets, test_size=0.2)
    pickle.dump(train_tweets, open('./data/tweets/train.p', 'wb'))
    pickle.dump(test_tweets, open('./data/tweets/test.p', 'wb'))


def load_training_test_sets():
    """Load training and test sets"""
    train = pickle.load(open('./data/tweets/train.p', 'rb'))
    test = pickle.load(open('./data/tweets/test.p', 'rb'))
    return train, test


def download_tweets():
    """Download Tweets using Tweepy"""
    secret = open('secret.txt')
    strs = secret.read().split("\n")

    consumer_key = strs[0]
    consumer_secret = strs[1]
    access = strs[2]
    access_secret = strs[3]

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access, access_secret)

    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    sentiment = open('./data/tweets/English_Twitter_sentiment.csv')
    all_ids = sentiment.read().split('\n')

    save_file = './data/tweets/examples.p'
    if path.exists(save_file):
        ids_to_examples = pickle.load(open(save_file, "rb" ))
    else:
        ids_to_examples = dict()

    ids_to_query = []

    done_count = 0
    for i in range(len(all_ids)):
        parts = all_ids[i].split(',')
        if len(parts) < 3:
            break
        id = parts[0]
        label = parts[1]
        if id not in ids_to_examples or ids_to_examples[id].text is None:
            ids_to_examples[id] = TweetTrainingExample(id=id, text=None, label=label)
            ids_to_query.append(id)
        else:
            done_count += 1

    print(str.format('Skipping the first {} tweets.', done_count))

    for i in range(0, len(ids_to_query), 100):
        statuses = api.statuses_lookup(ids_to_query[i:i+100], include_entities=[False * 100], trim_user=[True * 100])
        print("Number of Tweets downloaded: ", len(statuses))
        for status in statuses:
            id = status.id_str
            text = status.text
            ids_to_examples[id].text = text

        if i % 1000 == 0:
            # do a periodic save
            pickle.dump(ids_to_examples, open(save_file, "wb"))

    # save it all at the end
    pickle.dump(ids_to_examples, open(save_file, "wb"))


# p-value based on mcnemar's test
def __mcnemar_chi(b, c):
    chi = math.pow(abs(b - c), 2)/(b + c)
    return 1 - stats.chi2.cdf(chi, 1)


def calculate_mcnemars(y_none, y_with, y_test):
    """Calculate Mcnemar's given two datasets and "correct" set

    Args:
        y_none: Predictions without emoji vectors
        y_with: Predictions made with our emoji vectors
        y_test: True predictions

    Returns:

    """
    r = np.zeros(4)
    # syntax: 0=none/ours, 1=none/theirs | 0=no/no, 1=no/yes, 2=yes/no, 3=yes/yes

    for j in range(len(y_none)):
        ind = 0
        if y_none[j] == y_test[j]:
            ind += 2
        if y_with[j] == y_test[j]:
            ind += 1
        r[ind] += 1

    return __mcnemar_chi(r[2], r[1])


def prepare_tweet_vector_averages(tweets, p2v):
    """Take the vector sum of all tokens in each tweet

    Args:
        tweets: All tweets
        p2v: Phrase2Vec model

    Returns:
        Average vectors for each tweet
        Truth
    """
    tokenizer = tk.TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

    avg_vecs = list()
    y = list()

    for tweet in tweets:
        tokens = tokenizer.tokenize(tweet.text)
        avg_vecs.append(np.sum([p2v[x] for x in tokens], axis=0) / len(tokens))
        y.append(tweet.label)

    return avg_vecs, y


if __name__ == '__main__':
    if not path.exists('secret.txt'):
        print('You must provide secret.txt with Twitter auth tokens. See Tweepy API')
    else:
        download_tweets()
