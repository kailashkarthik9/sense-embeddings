import logging

import numpy as np
import plotly.graph_objects as go

from statistics_computer import StatisticsComputer
from utils import try_to_load_as_pickled_object_or_none
from word2vec_interface import Word2VecInterface, Dataset
from wordnet_interface import WordNetInterface

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')


class HypothesisValidator:
    """
    This class is used to validate the hypothesis on the correlation between word embeddings and senses
    """

    def __init__(self):
        self.wordnet_interface = WordNetInterface()
        self.word2vec_interface = Word2VecInterface(Dataset.GOOGLE_NEWS)
        self.statistics_computer = StatisticsComputer()
        self.vocabulary = self.get_vocabulary()
        # self.synsets = self.get_filtered_synsets()
        self.word_vectors = self.get_word_vectors()
        self.word_statistics = self.compute_word_statistics()

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
                if '_' not in word2vec_vocabulary[word2vec_index]:
                    common_vocabulary.add(wordnet_vocabulary[wordnet_index])
                wordnet_index += 1
                word2vec_index += 1
            elif word2vec_vocabulary[word2vec_index] > wordnet_vocabulary[wordnet_index]:
                wordnet_index += 1
            else:
                word2vec_index += 1
        logger.debug('Returning vocabulary of size ' + str(len(common_vocabulary)))
        return common_vocabulary

    def get_word_vectors(self):
        word_vectors = dict()
        for word in self.vocabulary:
            word_vectors[word] = self.word2vec_interface.get_word_vector(word)
        return word_vectors

    def get_filtered_synsets(self):
        filtered_synsets = dict()
        for synset_idx, synset in enumerate(self.wordnet_interface.get_all_synsets()):
            synset_dict = {
                'idx': synset_idx,
                'words': [word for word in self.wordnet_interface.get_synset_lemmas(synset) if word in self.vocabulary],
                'statistics': dict()
            }
            if synset_dict['words']:
                filtered_synsets[self.wordnet_interface.get_synset_id(synset)] = synset_dict
        return filtered_synsets

    def compute_synset_raw_statistics(self):
        for id, synset in self.synsets.items():
            synset['statistics'] = {
                'raw': {
                    'alpha': [],
                    'beta': []
                },
                'aggregated': {
                    'alpha': dict(),
                    'beta': dict()
                }
            }
            for first_word_idx in range(len(synset['words'])):
                for second_word_idx in range(first_word_idx):
                    first_word_vector = self.word_vectors[synset['words'][first_word_idx]]
                    second_word_vector = self.word_vectors[synset['words'][second_word_idx]]
                    alpha = self.statistics_computer.get_vector_distance(first_word_vector, second_word_vector)
                    synset['statistics']['raw']['alpha'].append(alpha)
                related_synset_idxs = [self.wordnet_interface.get_synset_id(s) for s in
                                       self.wordnet_interface.get_synsets(synset['words'][first_word_idx]) if
                                       self.wordnet_interface.get_synset_id(s) != id and
                                       self.wordnet_interface.get_synset_id(s) in self.synsets]
                for related_synset_idx in related_synset_idxs:
                    related_synset = self.synsets[related_synset_idx]
                    for synset_word in synset['words']:
                        for related_synset_word in related_synset['words']:
                            if synset_word == related_synset_word:
                                continue
                            beta = self.statistics_computer.get_vector_distance(self.word_vectors[synset_word],
                                                                                self.word_vectors[related_synset_word])
                            synset['statistics']['raw']['beta'].append(beta)

    def add_synset_aggregated_statistics(self):
        for synset in self.synsets.values():
            raw_alphas = synset['statistics']['raw']['alpha']
            raw_betas = synset['statistics']['raw']['beta']
            aggregated_statistics_alpha = synset['statistics']['aggregated']['alpha']
            aggregated_statistics_beta = synset['statistics']['aggregated']['beta']
            if raw_alphas:
                aggregated_statistics_alpha['mean'] = self.statistics_computer.get_vector_mean(raw_alphas)
                aggregated_statistics_alpha['max'] = self.statistics_computer.get_vector_max(raw_alphas)
                aggregated_statistics_alpha['min'] = self.statistics_computer.get_vector_min(raw_alphas)
                aggregated_statistics_alpha['range'] = self.statistics_computer.get_vector_range(raw_alphas)
                aggregated_statistics_alpha['std_dev'] = self.statistics_computer.get_vector_std_dev(raw_alphas)
            if raw_betas:
                aggregated_statistics_beta['mean'] = self.statistics_computer.get_vector_mean(raw_betas)
                aggregated_statistics_beta['max'] = self.statistics_computer.get_vector_max(raw_betas)
                aggregated_statistics_beta['min'] = self.statistics_computer.get_vector_min(raw_betas)
                aggregated_statistics_beta['range'] = self.statistics_computer.get_vector_range(raw_betas)
                aggregated_statistics_beta['std_dev'] = self.statistics_computer.get_vector_std_dev(raw_betas)

    def compute_word_statistics(self):
        word_map = dict()  # Key Each word - Value each participating synset
        for word_idx, word in enumerate(self.vocabulary):
            logger.debug(str(word_idx) + ' : ' + word)
            word_vector = self.word2vec_interface.get_word_vector(word)
            word_dict = {
                'synsets': dict(),
                'alpha': None,
                'beta': None,
                'representative_sense': None
            }
            word_map[word] = word_dict
            word_synsets = self.wordnet_interface.get_synsets(word)
            for synset_idx, synset in enumerate(word_synsets):
                synset_dict = {
                    'statistics': {
                        'raw': {
                            'alpha': list(),  # [ \alpha_{word, word'} ] where word' \in synset
                            'beta': list()  # [ [ \beta_{word, word''} ] ] where word'' \in related_synset
                        },
                        'aggregated': {  # Key aggregation function - Value
                            'alpha': float('inf'),  # Mean of raw alphas
                            'beta': list()  # Mean of each raw beta list
                        }
                    }
                }
                word_dict['synsets'][self.wordnet_interface.get_synset_id(synset)] = synset_dict

                # Compute Raw Alpha Values
                for word_in_synset in self.wordnet_interface.get_synset_lemmas(synset):
                    if word_in_synset != word and word_in_synset in self.vocabulary:
                        word_in_synset_vector = self.word2vec_interface.get_word_vector(word_in_synset)
                        synset_dict['statistics']['raw']['alpha'].append(
                            self.statistics_computer.get_vector_distance(word_vector, word_in_synset_vector))

                # Compute Raw Beta Values
                for related_synset_idx, related_synset in enumerate(word_synsets):
                    if related_synset_idx != synset_idx:
                        related_synset_betas = []
                        for word_in_related_synset in self.wordnet_interface.get_synset_lemmas(related_synset):
                            if word_in_related_synset != word and word_in_related_synset in self.vocabulary:
                                word_in_related_synset_vector = self.word2vec_interface.get_word_vector(
                                    word_in_related_synset)
                                related_synset_betas.append(
                                    self.statistics_computer.get_vector_distance(word_vector,
                                                                                 word_in_related_synset_vector))
                        if related_synset_betas:
                            synset_dict['statistics']['raw']['beta'].append(related_synset_betas)

                # Compute Aggregated Alpha and Beta Values
                synset_dict['statistics']['aggregated']['alpha'] = self.statistics_computer.get_vector_mean(
                    np.array(synset_dict['statistics']['raw']['alpha']))
                synset_dict['statistics']['aggregated']['beta'] = [
                    self.statistics_computer.get_vector_mean(np.array(synset_betas)) for synset_betas in
                    synset_dict['statistics']['raw']['beta']]

            word_alpha = float('inf')
            word_representative_sense = ''
            for synset_id, synset in word_dict['synsets'].items():
                if synset['statistics']['aggregated']['alpha'] < word_alpha:
                    word_alpha = synset['statistics']['aggregated']['alpha']
                    word_representative_sense = synset_id
            if word_alpha != float('inf'):
                word_dict['alpha'] = word_alpha
            else:
                logger.debug('Alpha omitted for ' + word)
            if word_representative_sense:
                if word_representative_sense in word_dict['synsets']:
                    word_betas = word_dict['synsets'][word_representative_sense]['statistics']['aggregated']['beta']
                    word_dict['beta'] = word_betas
                logger.debug('Beta omitted for ' + word)
                word_dict['representative_sense'] = word_representative_sense
            else:
                logger.debug('Beta and Representative Sense omitted for ' + word)
        return word_map

    def plot_statistics(self):
        y = np.array([
            word['alpha']
            for word in self.word_statistics.values()
            if word['alpha']
        ])
        x = np.array([i for i in range(len(y))])
        fig = go.Figure(data=go.Scattergl(x=x, y=y, mode='markers'))
        # fig.show()
        y = np.array([
            beta
            for word in self.word_statistics.values()
            if word['beta']
            for beta in word['beta']
        ])
        x = np.array([
            idx
            for idx, word in enumerate(self.word_statistics.values())
            if word['beta']
            for beta in word['beta']
        ])
        fig = go.Figure(data=go.Scattergl(x=x, y=y, mode='markers'))
        # fig.show()
        y = np.array([
            beta - word['alpha'] / beta
            for word in self.word_statistics.values()
            if word['beta'] and word['alpha']
            for beta in word['beta']
        ])
        x = np.array([i for i in range(len(y))])
        fig = go.Figure(data=go.Scattergl(x=x, y=y, mode='markers'))
        fig.show()


if __name__ == '__main__':
    # validator = HypothesisValidator()
    validator = try_to_load_as_pickled_object_or_none('data/validator.pkl')
    validator.plot_statistics()
    # save_as_pickled_object(validator, 'data/validator.pkl')
    print('pickle it!')
