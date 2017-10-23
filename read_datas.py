import pandas as pd
import numpy as np

# df = pd.read_csv("data/train.csv")

def fextract(data_frame):
    data_frame['len1'] = data_frame['question1'].str.len()
    data_frame['len2'] = data_frame['question2'].str.len()
    data_frame['question1'] = data_frame['question1'].str.lower()
    data_frame['question2'] = data_frame['question2'].str.lower()
    for i in range(26):
        character = chr(i+97)
        data_frame[character+'1'] = data_frame['question1'].str.count(character) #/ data_frame['len1']
        data_frame[character+'2'] = data_frame['question2'].str.count(character) #/ data_frame['len2']
    return data_frame

def lextract(data_frame):
    data_frame['true'] = data_frame['is_duplicate'] == 1
    data_frame['false'] = data_frame['is_duplicate'] == 0
    return data_frame

def preproceesing(data_frame):
    f_columns = []
    for i in range(26):
        f_columns.append(chr(i+97)+'1')
        f_columns.append(chr(i+97)+'2')
    f_columns.append('len1')
    f_columns.append('len2')
    # print f_columns

    feature_M = fextract(data_frame).as_matrix(columns=f_columns)
    labels_M = lextract(data_frame).as_matrix(columns=['true','false'])

    return feature_M, labels_M.astype(np.float32)

# print fextract(df)[0:1]
# print feature_M[0]
# print labels_M[0]
# print lextract(df)
# f, _ = preproceesing(df)
# print f
