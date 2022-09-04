from pathlib import Path
from math import exp
import numpy as np
import joblib
import logging
from functools import partial
from nltk.util import ngrams
from collections import Counter, defaultdict


def load_file_corpora(dataset_path: str):
    """ This function yields a list of tokens for each sentence in the dataset
    This function works for corpora with many .txt files in the root directory.
    """
    for p in Path(dataset_path).glob("*.txt"):
        document = open(p, "r", encoding="utf8").readlines()
        for sent in document:
            yield sent.strip("\n").split(" ")           

class NgramLanguageModel(object):

    def __init__(self, text_loader=None, model: tuple = None):
        """ The Constructor builds the language model.
        :param text_loader: A generator that yields lists of tokens. Leave blank if loading a model from :param model:
        :param model: Load the model given here as a tuple of (vocabulary, inverse_vocabulary, ngram_probabilities).
                      Leave blank if building model from texts
        """
        
        if model:
            self.vocabulary = model[0]
            self.inverse_vocabulary = model[1]
            self.four_gram_probabilities = model[2]
            return

        logging.info("build vocabulary")
        logging.info("build n-gram count matrices")
        self.four_gram_counts = self._count_ngrams(text_loader)
        logging.info("calculate conditional probabilities")
        self.four_gram_probabilities = self._build_model(self.four_gram_counts)
        
    @staticmethod
    def load(name):
        """ Call this to load a prepared model from storage:
         """
        store_at = Path(f"models/{name}")
        logging.info(f"load model from {store_at}")
        four_gram_counts = joblib.load(open(store_at / "ngram_counts.pkl", 'rb'))
        four_gram_probabilities = joblib.load(open(store_at / "probabilities.pkl", 'rb'))
        return NgramLanguageModel(model=(four_gram_counts, four_gram_probabilities))

    def _count_ngrams(self, sequences) -> tuple:
        """ This function counts, how often a word appears in the corpus yielded by sequences and how often a word
            appears as the last word in an n-gram.

        :param sequences: a generator that yields a list of words ["<s>", "as", "the", ...]
        :return: ngram_counts,
        """
        # How often each word w_n occurs after a certain four-gram.
        four_gram_counts = defaultdict(lambda: defaultdict(lambda: 0))                                      # Using default dictionary to create a matrix to store four_gram counts
        for sequence in sequences():
            # Using ngrams from nltk to create four_grams
            for w1, w2, w3, w4 in ngrams(sequence, 4, pad_right=True, pad_left=True):                   # Applying padding on both left and right side because of four_grams
                four_gram_counts[(w1, w2, w3)][w4] += 1                                                 # Storing counts of four grams
        return four_gram_counts


    def _build_model(self, four_gram_counts):
        """ estimate the conditional probabilities P(w_i | w_i-1, ..., w_i-n) of each n-gram.
        :param ngram_counts
        :return: probabilities of four_grams
        """
        four_gram_probabilities = defaultdict(lambda: defaultdict(lambda: 0))
        # Calculating probabilities
        for w1w2w3 in four_gram_counts:
            tri_count = float(sum(four_gram_counts[w1w2w3].values()))               # sum of all the counts for tri-grams (w1,w2,w3)
            for w4 in four_gram_counts[w1w2w3]:
                four_gram_probabilities[w1w2w3][w4] = four_gram_counts[w1w2w3][w4]/tri_count         # number of occurences of w4 after w1, w2 and w3
        return four_gram_probabilities

    def _get_pr_or_0(self, tokens):
        """ Get the conditional probability of observing the word
        """
        for w1w2w3 in self.four_gram_counts:
            for w4 in self.four_gram_counts[w1w2w3]:
                yield self.four_gram_probabilities[w1w2w3][w4]

    def perplexity_of(self, tokens):
        """ Compute the perplexity of the LM on the tokens """
        four_gram_counts =self._count_ngrams(tokens)                                     # getting four_gram_counts for the test data
        four_gram_probabilities_test = self._build_model(four_gram_counts)               # getting the four_gram_probabilites for the test data
        probabilities = list(self._get_pr_or_0(tokens))
        sequence_probability = np.sum(np.log2(probabilities))
        perplexity = np.power(-(1 / len(probabilities)) * sequence_probability, 2)       # getting the perplexity by applying the formula given in the suplement file
        return perplexity, four_gram_probabilities_test                                  # returning four_gram_probabilities_test so that it can be used in complete_sequence_after

    def complete_sequence_after(self,tokens, four_gram_probabilities):
        """ Predict the next following words until <eos> is reached. Note that we
        :param tokens: a list of words and four gram probabilities
        """
        while tokens[-3] != '<eos>':                                                     # Checking if the last token is not the end of sentence
            next_word_prob = 0
            r = 0.3                                                                      # Creating a random threshold to select the probability value
            for next_word in four_gram_probabilities[tuple(tokens[-3:])].keys():
                next_word_prob += four_gram_probabilities[tuple(tokens[-3:])][next_word]
                if next_word_prob >= r:                                                  # only those words are selected that are above the randomly selected threshold
                    tokens.append(next_word)
                    break
        return " ".join([word for word in tokens if word])


if __name__ == "__main__":
    # Build the language model from the text sources.
    text_train = (partial(load_file_corpora, Path("lm-dev")))
    lm = NgramLanguageModel(text_train)
    four_gram_probabilities = lm.four_gram_probabilities
    # Calculate the perplexity on the test set. Lower is better.
    text = (partial(load_file_corpora, Path("lm-test")))
    perplexity, four_gram_probabilities_test = lm.perplexity_of(text)
    print("text perplexity:", perplexity)

    # Complete the sentence based on the most likely next tokens.
    # <s> the year 1866 was signalised by a remarkable incident , a mysterious and puzzling phenomenon ,
    #   which doubtless no one has yet forgotten . <eos>
    print(lm.complete_sequence_after(["<s>", "the", "way"], four_gram_probabilities))
    print(lm.complete_sequence_after(["<s>", "the", "year", "1866", "was"], four_gram_probabilities_test))

