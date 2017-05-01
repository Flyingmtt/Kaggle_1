# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 21:24:49 2017

@author: HeSijia
"""

from time import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import matplotlib.pyplot as plt
#import seaborn as sns
import nltk
import re
from sklearn.metrics import log_loss

from nltk.corpus import stopwords
from nltk import SnowballStemmer

SnowballStemmer.languages # See which languages are supported
ps = nltk.stem.snowball.EnglishStemmer()

#pal = sns.color_palette()

df_train=pd.read_csv('/Users/MTT/Desktop/MTT/kaggle/quora/data/train.csv')
#df_train=df_train.head(20000)
data=pd.DataFrame()
#df_test=pd.read_csv('test.csv')

stops = set(stopwords.words("english"))
punc=['?','(',')',':',',','.','!',"'",'"','%']

t0 = time()
#这里放你要计算时间的代码#

def drop_stopwords(tokens):
    final_tokens=set()
    for word in tokens:
        if word not in stops:
            final_tokens.add(word)
    return final_tokens


def zzbds(sent):
    sent = re.sub(r"what's", "what is", sent)
    sent = re.sub(r"\'s", "", sent)
    sent = re.sub(r"\'ve", " have", sent)
    sent = re.sub(r"can't", "cannot", sent)
    sent = re.sub(r"n't", " not", sent)
    sent = re.sub(r"i'm", "i am", sent)
    sent = re.sub(r"\'re", " are", sent)
    sent = re.sub(r"\'d", " would", sent)
    sent = re.sub(r"\'ll", " will", sent)
    sent = re.sub(r"\^", " ^ ", sent)
    sent = re.sub(r"\+", " + ", sent)
    sent = re.sub(r"\-", " - ", sent)
    sent = re.sub(r"\=", " = ", sent)
    sent = re.sub(r"(\d+)(k)", r"\g<1>000", sent)
    sent = re.sub(r" e.g ", " eg ", sent)
    sent = re.sub(r" b.g ", " bg ", sent)
    sent = re.sub(r" u.s ", " american ", sent)
    sent = re.sub(r"\0s", "0", sent)
    sent = re.sub(r" 9 11 ", "911", sent)
    sent = re.sub(r"e - mail", "email", sent)
    sent = re.sub(r"j k", "jk", sent)
    tokens=[n for n in re.findall(r"[^:?!(),.' \/\\\[\]]+",sent)]
    
    tokens=drop_stopwords(tokens)
    return tokens

def find_noun(sent):
    tokens=[n for n in sent]
    tags=nltk.pos_tag(tokens)
    a=[b[0] for b in tags if b[1] in ['NN','NNS','NNP','NNPS']]

    return len(a)

def calculate_diff1(row):
    R = -(len(row['q1_un'])+len(row['q2_un']))/(len(row['common_words'])+1)
    return R

def calculate_diff2(row):
    R = -(len(row['q1_un'])+len(row['q2_un']))
    return R

def calculate_diff3(row):
    R = -(row['count_nn1']+row['count_nn2'])/(len(row['common_words'])+1)
    return R

def calculate_diff4(row):
    R = -(row['count_nn1']+row['count_nn2'])
    return R

def comp_first(row):
    a=str(row['first_1'])
    b=str(row['first_2'])
    if a != b:
        if (a == 'what') and (b == 'how ') or (a == 'how ') and (b == 'what'):
            comp_result=0.5
        else:
            comp_result=0
    else:
        comp_result=1
    return comp_result


data.loc[:,'id']=df_train.iloc[:,0]
data.loc[:,'is_duplicate']=df_train.iloc[:,5]
data.loc[:,'question1']=df_train.question1.apply(lambda x:str(x).lower())
data.loc[:,'question2']=df_train.question2.apply(lambda x:str(x).lower())
data.loc[:,'token_q1']=data.question1.apply(zzbds)
data.loc[:,'token_q2']=data.question2.apply(zzbds)
data.loc[:,'common_words']=data.apply(
        lambda x: set(x['token_q1']).intersection(set(x['token_q2'])),axis=1)
data.loc[:,'q1_un']=data.apply(
        lambda x:set(x['token_q1']).difference(set(x['common_words'])),axis=1)
data.loc[:,'q2_un']=data.apply(
        lambda x:set(x['token_q2']).difference(set(x['common_words'])),axis=1)
data.loc[:,'count_nn1']=data.q1_un.apply(lambda x:find_noun(x))
data.loc[:,'count_nn2']=data.q2_un.apply(lambda x:find_noun(x))
data.loc[:,'cal_diff1']=data.apply(calculate_diff1,axis=1)
data.loc[:,'cal_diff2']=data.apply(calculate_diff2,axis=1)
data.loc[:,'cal_diff3']=data.apply(calculate_diff3,axis=1)
data.loc[:,'cal_diff4']=data.apply(calculate_diff4,axis=1)
data.loc[:,'first_1']=data.question1.apply(lambda x:str(x)[0:4])
data.loc[:,'first_2']=data.question2.apply(lambda x:str(x)[0:4])
data.loc[:,'comp_first_re']=data.apply(comp_first,axis=1)


#plt.figure(figsize=(15, 5))
train_word_match2 = data.loc[:,'cal_diff1']
#train_word_match2 = data.loc[:,'comp_first_re']
#plt.hist(train_word_match2[df_train['is_duplicate'] == 0].dropna(),
#         bins=20, normed=True, label='Not Duplicate')
#plt.hist(train_word_match2[df_train['is_duplicate'] == 1].dropna(),
#         bins=20, normed=True, alpha=0.7, label='Duplicate')
#plt.legend()
#plt.title('Label distribution over word_match_share', fontsize=15)
#plt.xlabel('word_match_share', fontsize=15)



print("training time:", round(time()-t0, 3), "s")

from sklearn.metrics import roc_auc_score
print('Original AUC:', roc_auc_score(
        df_train['is_duplicate'], train_word_match2))
