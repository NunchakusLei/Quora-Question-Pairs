import pandas as pd
import numpy as np
import time
import string
from word2vec import *

# df = pd.read_csv("data/train.csv")
# feature_filename = "data/features_mini_test.csv"
feature_filename = "data/features.csv"

def fextract(data_frame):
    # data_frame['len1'] = data_frame['question1'].str.len()
    # data_frame['len2'] = data_frame['question2'].str.len()
    # data_frame['question1'] = data_frame['question1'].str.lower()
    # data_frame['question2'] = data_frame['question2'].str.lower()
    # for i in range(26):
    #     character = chr(i+97)
    #     data_frame[character+'1'] = data_frame['question1'].str.count(character) #/ data_frame['len1']
    #     data_frame[character+'2'] = data_frame['question2'].str.count(character) #/ data_frame['len2']

    def extract_sentence_vec(col_name):
        q_vecs = None
        progress = 0
        for row in data_frame[col_name]:
            if type(row)!=type(str()):
                row = ''
            temp_sv = sentence_vec(row.translate(None, string.punctuation))
            if q_vecs is None:
                q_vecs = temp_sv
            else:
                q_vecs = np.vstack((q_vecs, temp_sv))
            progress += 1
            if progress%1000 == 0:
                print "Extracted %d rows for %s." % (progress, col_name)
        return q_vecs


    q1_vecs = extract_sentence_vec('question1')
    q2_vecs = extract_sentence_vec('question2')
    cos_similarities = cos_similarity_matrix(q1_vecs, q2_vecs)

    # store q_vecs into data frame
    for i in range(q1_vecs.shape[1]):
        data_frame['q1_vec_col'+str(i)] = q1_vecs[:,i]
        data_frame['q2_vec_col'+str(i)] = q2_vecs[:,i]

    data_frame['cos_similarity'] = cos_similarities

    return data_frame

def lextract(data_frame):
    data_frame['true'] = data_frame['is_duplicate'] == 1
    data_frame['false'] = data_frame['is_duplicate'] == 0
    return data_frame

def preproceesing(data_frame):
    f_columns = []
    # for i in range(26):
    #     f_columns.append(chr(i+97)+'1')
    #     f_columns.append(chr(i+97)+'2')
    # f_columns.append('len1')
    # f_columns.append('len2')
    # print f_columns

    # feature_M = load_feature("data/quora_features_train_with_truth.csv")
    feature_M = load_feature()
    if feature_M is None:
        feature_M = fextract(data_frame)
        save_feature(feature_M)

    labels_M = data_frame.as_matrix(columns=['is_duplicate'])

    # print "I'm here."
    # print feature_M.columns.tolist()
    f_columns = feature_M.columns.tolist()
    no_feature_col = ['id','qid1','qid2','question1','question2','is_duplicate']
    for name in no_feature_col:
        f_columns.pop(f_columns.index(name))
    # f_columns.pop(f_columns.index('is_duplicate'))
    feature_M = feature_M.as_matrix(columns=f_columns)

    # matrix1 = np.array(list(data_frame['q1_sv']))
    # matrix2 = np.array(list(data_frame['q2_sv']))
    # feature_M = np.hstack((matrix1, matrix2))
    # feature_M = feature_M.as_matrix(columns=['cos_similarity'])
    # labels_M = lextract(data_frame).as_matrix(columns=['true','false'])

    # return feature_M, labels_M.astype(np.float32)
    return feature_M, labels_M

def load_feature(filename=feature_filename):
    try:
        df = pd.read_csv(filename)
    except IOError:
        df = None
    return df

def save_feature(data_frame, filename=feature_filename):
    data_frame.to_csv(filename, encoding='utf-8', index=False)

if __name__ == "__main__":
    """
    Main function for testing
    """
    start_time = time.time()
    df = load_feature("data/train_mini_test.csv")
    end_time = time.time()
    print "panda open file time: %fs" % (end_time-start_time)

    start_time = time.time()
    preproceesing(df)
    end_time = time.time()
    print "preproceesing time: %fs" % (end_time-start_time)
