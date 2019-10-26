from nltk.corpus import wordnet


class WordNetInterface:
    """
    This class acts as the interface to the external wordnet library. All the methods used to access wordnet resources
    are encapsulated here.
    """

    @staticmethod
    def example_wordnet():
        """
        This is an example method on how to use wordnet to get synsets and illustrates the attributes of potential interest
        :return: None
        """
        synsets_example = wordnet.synsets('example')
        for synset in synsets_example:
            print('POS : ' + synset.pos())
            print('Definition : ' + synset.definition())
            for lemma_idx, lemma in enumerate(synset.lemmas()):
                print(str(lemma_idx + 1) + '. ' + lemma.name())
            print('')

    @staticmethod
    def get_synsets(word: str):
        """
        This method returns the synsets of a word
        :param word: The word whose synsets are to be retrieved
        :return: List of Synset instances
        """
        return wordnet.synsets(word)

    @staticmethod
    def get_synset_id(synset: wordnet.Synset):
        """
        This method gets a unique identifier for a synset
        :param synset: The synset whose identifier is to be retrieved
        :return: A string identifier
        """
        return synset.name()

    @staticmethod
    def get_synset_by_id(id: str):
        """
        This method gets the synset given an id
        :param id: The id for which the synset is to be retrieved
        :return: The Synset object
        """
        return wordnet.synset(id)

    @staticmethod
    def get_synset_pos(synset):
        """
        This method retrieves the part of speech associated with the synset
        :param synset: The synset whose POS is to be retrieved
        :return: A character representing the POS of the synset
        """
        return synset.pos()

    @staticmethod
    def get_synset_definition(synset):
        """
        This method retrieves the definition of the synset - a description of what I would think of as a semantic concept
        :param synset: The synset whose description is to be retrieved
        :return: A string description
        """
        return synset.definition()

    @staticmethod
    def get_synset_lemmas(synset):
        """
        This method gets all the words in the synset
        :param synset: The synset whose members are to be retrieved
        :return: A list of words composing the synset
        """
        return [lemma.name() for lemma in synset.lemmas()]

    @staticmethod
    def get_vocabulary():
        """
        This method returns the vocabulary of wordnet
        :return: List of words in the wordnet vocabulary
        """
        return sorted(wordnet.words())


if __name__ == '__main__':
    validator = WordNetInterface()
    validator.example_wordnet()
