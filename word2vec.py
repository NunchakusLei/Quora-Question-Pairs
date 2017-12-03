import pandas as pd
import numpy as np
import time

from gensim.models import KeyedVectors

filename = 'data/GoogleNews-vectors-negative300.bin'
wv_model = KeyedVectors.load_word2vec_format(filename, binary=True)

def sentence_vec(question, model=wv_model):
    result = np.zeros(model['man'].shape)
    for word in question.split():
        if word in model.vocab:
            result += model[word]
    return result

def cos_similarity(vec1, vec2):
    denominator = np.dot(vec1, vec2)
    numerator = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return denominator / numerator

def cos_similarity_matrix(m1, m2):
    denominator = np.sum(m1*m2, axis=1)
    numerator = np.linalg.norm(m1,axis=1) * np.linalg.norm(m2,axis=1)
    return list(denominator / numerator)

if __name__ == "__main__":
    """
    Main function for testing
    """
    q1 = 'What is the step by step guide to invest in share market in india'
    q2 = 'What is the step by step guide to invest in share market'
    print sentence_vec(wv_model, q1)
