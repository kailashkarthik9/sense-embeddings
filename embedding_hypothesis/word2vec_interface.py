from enum import Enum

from gensim.models import KeyedVectors


class Dataset(Enum):
    """
    This enumeration details all the possible sources to load word2vec embeddings from.
    Download the corresponding models and place them in the /data directory
    """
    GOOGLE_NEWS = 'GoogleNews-vectors-negative300.bin'
    TWITTER = 'word2vec_twitter_tokens'
    # WIKIPEDIA = ''


class Word2VecInterface:
    """
    This class acts as the interface to the external word2vec embeddings. All the methods used to access word2vec
    are encapsulated here. Word2vec is accessed through Gensim
    """

    def __init__(self, dataset: Dataset):
        self.model = KeyedVectors.load_word2vec_format('data/' + dataset.value, binary=True)

    def get_word_vector(self, word):
        """
        This method returns the embedding for a given word
        :param word: The word for which embedding is to be retrieved
        :return: N dimensional vector
        """
        return self.model[word]

    def get_vocabulary(self):
        """
        This method returns the vocabulary of the word2vec model
        :return: A list of words in the vocabulary
        """
        return sorted(self.model.vocab)
