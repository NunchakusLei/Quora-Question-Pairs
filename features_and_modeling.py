
# coding: utf-8

# In[4]:

import pandas as pd
import numpy as np
import nltk
import gensim 
import pyemd
import pickle
import matplotlib.pyplot as plt
import os
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize
from gensim.models import KeyedVectors
if not os.path.exists('./GoogleNews-vectors-negative300.bin.gz'):
    raise ValueError("SKIP: You need to download the google news model")
from gensim.similarities import WmdSimilarity
stop_words = stopwords.words('english')
import _pickle as cPickle


# In[5]:

f = pd.read_csv('/Users/joe/Desktop/data/train.csv')
test_f=pd.read_csv('/Users/joe/Desktop/data/test.csv')


# In[6]:

df = pd.DataFrame(f)
data_test= pd.DataFrame(test_f)


# In[331]:

test_f


# In[7]:

data = f.drop(['id', 'qid1', 'qid2'], axis=1)


# In[8]:

def feature1(data):
    data['len_q1'] = data.question1.apply(lambda x: len(str(x)))
    data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
    data['diff_len'] = data.len_q1 - data.len_q2
    data['len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
    data['len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
    data['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))
    data['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))
    data['common_words'] = data.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
    data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
    data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)


# In[9]:

feature1(data)


# In[10]:

feature1(data_test)


# In[15]:

data


# In[16]:

data_test


# In[13]:

# data['len_q1'] = data.question1.apply(lambda x: len(str(x)))
# data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
# data['diff_len'] = data.len_q1 - data.len_q2
# data['len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
# data['len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
# data['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))
# data['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))
# data['common_words'] = data.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
# data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
# data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
# data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
# data['fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
# data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
# data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
# data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)


# In[27]:

def wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s1 = [w for w in s1 if w.isalpha()]
    s2 = [w for w in s2 if w not in stop_words]
    s2 = [w for w in s2 if w.isalpha()]
    return model.wmdistance(s1, s2)

def norm_wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s1 = [w for w in s1 if w.isalpha()]
    s2 = [w for w in s2 if w not in stop_words]
    s2 = [w for w in s2 if w.isalpha()]
    return norm_model.wmdistance(s1, s2)
def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())
def w2v(data, norm):
    if (norm):
        norm_model.init_sims(replace=True)
        data['norm_wmd'] = data.apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)
    else:
        data['w2v'] = data.apply(lambda x: wmd(str(x['question1']), str(x['question2'])), axis=1)
    return data


# In[28]:

model= KeyedVectors.load_word2vec_format('/Users/joe/Desktop/GoogleNews-vectors-negative300.bin.gz', binary=True)


# In[29]:

norm_model = gensim.models.KeyedVectors.load_word2vec_format('/Users/joe/Desktop/GoogleNews-vectors-negative300.bin.gz', binary=True)


# In[30]:

w2v(data,True)


# In[31]:

w2v(data,False)


# In[32]:

w2v(data_test,True)


# In[34]:

w2v(data_test,False)


# In[10]:

# model= KeyedVectors.load_word2vec_format('/Users/joe/Desktop/GoogleNews-vectors-negative300.bin.gz', binary=True)
# model.init_sims(replace=True)
# data['w2v'] = data.apply(lambda x: wmd(str(x['question1']), str(x['question2'])), axis=1)


# In[23]:

# norm_model = gensim.models.KeyedVectors.load_word2vec_format('/Users/joe/Desktop/GoogleNews-vectors-negative300.bin.gz', binary=True)
# norm_model.init_sims(replace=True)
# data['norm_wmd'] = data.apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)


# In[36]:

def feature3(data):
    question1_vectors = np.zeros((data.shape[0], 300))
    error_count = 0
    for i, q in tqdm(enumerate(data.question1.values)):
        question1_vectors[i, :] = sent2vec(q)

    question2_vectors  = np.zeros((data.shape[0], 300))
    for i, q in tqdm(enumerate(data.question2.values)):
        question2_vectors[i, :] = sent2vec(q)

    data['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    data['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]
    data['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    data['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]  
    data['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

    data['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]
    data['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

    data['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
    data['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
    data['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
    data['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]
    return data


# In[37]:

feature3(data)


# In[43]:

feature3(data_test)


# In[44]:

data_test


# In[45]:

data.to_csv('data/train_W2v_checkpoint.csv', index=False)
data_test.to_csv('data/test_W2v_checkpoint.csv', index=False)


# In[168]:

from collections import defaultdict
ques = pd.concat([f[['question1', 'question2']],
        test_f[['question1', 'question2']]], axis=0).reset_index(drop='index')
ques.shape


# In[169]:

q_dict = defaultdict(set)
for i in range(ques.shape[0]):
        q_dict[ques.question1[i]].add(ques.question2[i])
        q_dict[ques.question2[i]].add(ques.question1[i])


# In[171]:

def q1_q2_intersect(row):
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))


# In[179]:

data['q1_q2_intersect'] = f.apply(q1_q2_intersect, axis=1, raw=True)
data_test['q1_q2_intersect'] = test_f.apply(q1_q2_intersect, axis=1, raw=True)


# In[304]:

stops = set(stopwords.words("english"))

def word_match_share(q1, q2, stops=None):
    q1 = str(q1).lower().split()
    q2 = str(q2).lower().split()
    q1words = {}
    q2words = {}
    for word in q1:
        if word not in stops:
            q1words[word] = 1
    for word in q2:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0.
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R


# In[305]:

q_dict = defaultdict(dict)
for i in range(ques.shape[0]):
        wm = word_match_share(ques.question1[i], ques.question2[i], stops=stops)
        q_dict[ques.question1[i]][ques.question2[i]] = wm
        q_dict[ques.question2[i]][ques.question1[i]] = wm


# In[307]:

def q1_q2_wm_ratio(row):
    q1 = q_dict[row['question1']]
    q2 = q_dict[row['question2']]
    inter_keys = set(q1.keys()).intersection(set(q2.keys()))
    if(len(inter_keys) == 0): return 0.
    inter_wm = 0.
    total_wm = 0.
    for q,wm in q1.items():
        if q in inter_keys:
            inter_wm += wm
        total_wm += wm
    for q,wm in q2.items():
        if q in inter_keys:
            inter_wm += wm
        total_wm += wm
    if(total_wm == 0.): return 0.
    return inter_wm/total_wm


# In[308]:

data['q1_q2_wm_ratio'] = f.apply(q1_q2_wm_ratio, axis=1, raw=True)
data_test['q1_q2_wm_ratio'] = test_f.apply(q1_q2_wm_ratio, axis=1, raw=True)


# In[332]:

dfs = (f, test_f)


# In[333]:

dfs


# In[ ]:

# question1_vectors = np.zeros((data.shape[0], 300))
# error_count = 0


# In[24]:

# for i, q in tqdm(enumerate(data.question1.values)):
#     question1_vectors[i, :] = sent2vec(q)

# question2_vectors  = np.zeros((data.shape[0], 300))
# for i, q in tqdm(enumerate(data.question2.values)):
#     question2_vectors[i, :] = sent2vec(q)


# In[25]:

# data['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
#                                                           np.nan_to_num(question2_vectors))]

# data['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
#                                                           np.nan_to_num(question2_vectors))]


# In[26]:

# data['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
#                                                           np.nan_to_num(question2_vectors))]

# data['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
#                                                           np.nan_to_num(question2_vectors))]


# In[27]:

# data['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
#                                                           np.nan_to_num(question2_vectors))]

# data['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
#                                                           np.nan_to_num(question2_vectors))]


# In[28]:

# data['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
#                                                           np.nan_to_num(question2_vectors))]

# data['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]


# In[29]:

# data['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
# data['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
# data['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]


# In[30]:

# cPickle.dump(question1_vectors, open('dat.pkl', 'wb'), -1)
# cPickle.dump(question2_vectors, open('data/q2_w2v.pkl', 'wb'), -1)


# In[47]:

# data=data.drop(['question1', 'question2','is_duplicate','lower_1','how1','where1','why1','when1','who1','what1','lower_2','how2','where2','why2','when2','who2','what2','prossesdata','bi','diff','phrased','phrased2'], axis=1)
# data.to_csv('data/quora_features.csv', index=False)


# In[53]:

data=data.drop(['is_duplicate'], axis=1)
data_test= data_test.drop(['test_id','question1', 'question2'], axis=1)


# In[312]:

data.to_csv('data/quora_features_train.csv', index=False)
data_test.to_csv('data/quora_features_test.csv', index=False)


# In[76]:

# data.to_csv('data/quora_features.csv', index=False)


# In[313]:

#df_train = pd.read_csv('/Users/joe/Desktop/811project/test.csv')

X_train = pd.read_csv('data/quora_features_train.csv', index_col=False)
#X_train.head()
X_test = pd.read_csv('data/quora_features_test.csv', index_col=False)


# In[314]:

y_train = f['is_duplicate'].values


# In[315]:

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing


# In[316]:

X_train = X_train.replace(np.nan, 0.)
X_train = X_train.replace(np.inf, 99.)
X_test = X_test.replace(np.nan, 0.)
X_test = X_test.replace(np.inf, 99.)


# In[269]:

#X_train = preprocessing.scale(X_train)


# In[317]:

X_train.isnull().values.any()


# In[318]:

X_test.isnull().values.any()


# In[98]:

# X_train.index.values


# In[285]:

#re1 = lr.predict(X_train[:])
#re1
def check_rate(orginal, result_prob):

    temp = pd.DataFrame(columns=['test_id','is_duplicate_truth','result'])
    #result_df.drop(['is_duplicate'], axis=1)
    temp['test_id']= orginal.index.values
    temp['is_duplicate_truth']=result_prob[:]
    temp['result']=orginal['is_duplicate']
    temp['is_it_correct'] = np.where(temp['is_duplicate_truth']==temp['result'], 1, 0)
    correct=0;
    for i in temp['is_it_correct']:
        correct= i +correct
    return correct/404290


# In[287]:

lr = LogisticRegression()
lr.fit(X_train, y_train)
#lr= LogisticRegression(solver='liblinear').fit(X_train, y_train)
re1 = lr.predict_proba(X_test[:])
#check_rate(f,re1)


# In[288]:

re1


# In[289]:

result_df = pd.DataFrame(columns=['test_id','is_duplicate'])
#result_df.drop(['is_duplicate'], axis=1)
result_df['test_id']= test_f.index.values
result_df['is_duplicate']=re1[:,1]


# In[290]:

result_df


# In[291]:

result_df.to_csv('data/LogisticRegression.csv', index=False)


# In[319]:

from sklearn.ensemble import GradientBoostingClassifier


# In[320]:

gb= GradientBoostingClassifier()
gb


# In[322]:

gb= gb.fit(X_train, y_train)
gb1 = gb.predict_proba(X_test[:])
#check_rate(f,gb1)


# In[324]:

gb1[:30]


# In[325]:

result_dfg = pd.DataFrame(columns=['test_id','is_duplicate'])
#result_df.drop(['is_duplicate'], axis=1)
result_dfg['test_id']= test_f.index.values
result_dfg['is_duplicate']=gb1[:,1]
result_dfg.to_csv('data/GradientBoostingClassifier2.csv', index=False)


# In[297]:

result_dfg


# In[ ]:

#fist model test end here


# In[89]:

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
re2 = rf.predict_proba(X_train[:])
re


# In[91]:

result_df2 = pd.DataFrame(columns=['test_id','is_duplicate'])
#result_df.drop(['is_duplicate'], axis=1)
result_df2['test_id']= test_f.index.values
result_df2['is_duplicate']=re2[:,1]
result_df2.to_csv('data/RandomForestClassifier.csv', index=False)


# In[93]:

from sklearn.neighbors import NearestNeighbors


# In[103]:

kn = NearestNeighbors()
#kn.fit(X_train, y_train)
kn=NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X_train, y_train)
kn2 = rf.predict_proba(X_test[:])
kn2


# In[96]:

result_df3 = pd.DataFrame(columns=['test_id','is_duplicate'])
#result_df.drop(['is_duplicate'], axis=1)
result_df3['test_id']= test_f.index.values
result_df3['is_duplicate']=kn2[:,1]
result_df3.to_csv('data/NearestNeighbors.csv', index=False)


# In[97]:

result_df3


# In[ ]:



