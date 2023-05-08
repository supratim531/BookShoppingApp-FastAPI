import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer


# nltk.download("punkt")


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word=""):
    stemmer = PorterStemmer()
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    tokenized_sentence = ['Is', 'anyone', 'organize', 'this', 'program', '?']
    words = ['are', 'you', 'organ', 'thi', 'program']
    bag = [0, 0, 1, 1, 1]
    """

    # initialize numpy_array(bag) with 0 of length words
    bag = np.zeros(len(words), dtype=np.float32)
    stemmed_tokenized_sentence = [stem(word) for word in tokenized_sentence]

    # prepare bag
    for (index, word) in enumerate(words):
        if word in stemmed_tokenized_sentence:
            bag[index] = 1.0
    return bag


if __name__ == "__main__":
    # print(tokenize("Hello! I'm a big fan of yours..."))
    # print(stem("Beautiful"), [stem(w) for w in ["Anyone", "University"]])
    # print(bag_of_words(['Is', 'anyone', 'organize', 'this', 'program', '?'], ['are', 'you', 'organ', 'thi', 'program']))
    pass
