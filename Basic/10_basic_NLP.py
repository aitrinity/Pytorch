import numpy as np 
import nltk 
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


# Helper function!
def load_data(filename):

    with open(filename, 'r') as reader:
    # Read & print the entire file
        data = reader.read()
    
    return data

def tokenize(sentence):

    return nltk.word_tokenize(sentence)

def stem(word):

    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):

    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag

# Load the data

data = load_data('test.txt')

# Tokenize the data
data_tokenized = tokenize(data)

# Stem the data
data_stem = [stem(word) for word in data_tokenized]

print(data_stem)