from statistics_computer import StatisticsComputer
from word2vec_interface import Word2VecInterface, Dataset
from wordnet_interface import WordNetInterface


class HypothesisValidator:
    """
    This class is used to validate the hypothesis on the correlation between word embeddings and senses
    """

    def __init__(self):
        self.wordnet_interface = WordNetInterface()
        self.word2vec_interface = Word2VecInterface(Dataset.GOOGLE_NEWS)
        self.statistics_computer = StatisticsComputer()
        self.vocabulary = self.get_vocabulary()
        self.synsets = self.get_filtered_synsets()

    def get_vocabulary(self):
        wordnet_vocabulary = self.wordnet_interface.get_vocabulary()
        word2vec_vocabulary = self.word2vec_interface.get_vocabulary()
        common_vocabulary = set()
        wordnet_vocabulary_size = len(wordnet_vocabulary)
        word2vec_vocabulary_size = len(word2vec_vocabulary)
        wordnet_index = 0
        word2vec_index = 0
        while word2vec_index < word2vec_vocabulary_size and wordnet_index < wordnet_vocabulary_size:
            if word2vec_vocabulary[word2vec_index] == wordnet_vocabulary[wordnet_index]:
                common_vocabulary.add(wordnet_vocabulary[wordnet_index])
                wordnet_index += 1
                word2vec_index += 1
            elif word2vec_vocabulary[word2vec_index] > wordnet_vocabulary[wordnet_index]:
                wordnet_index += 1
            else:
                word2vec_index += 1
        return common_vocabulary

    def get_filtered_synsets(self):
        synsets = dict()
        for word in self.vocabulary:
            word_vector = self.word2vec_interface.get_word_vector(word)
            word_dict = dict()
            synsets[word] = word_dict
            for synset in self.wordnet_interface.get_synsets(word):
                synset_dict = dict()
                word_dict[self.wordnet_interface.get_synset_id(synset)] = synset_dict
                for synonym in self.wordnet_interface.get_synset_lemmas(synset):
                    if synonym in self.vocabulary:
                        synonym_vector = self.word2vec_interface.get_word_vector(synonym)
                        synset_dict[synonym] = self.statistics_computer.get_vector_distance(word_vector, synonym_vector)
        return synsets
