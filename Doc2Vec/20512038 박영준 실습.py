# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 16:17:25 2020

@author: young
"""
from konlpy.tag import Hannanum
from pprint import pprint
import nltk

from konlpy.tag import Twitter
import pandas as pd

kor_corpus = pd.read_csv('news_koreanwave.csv', encoding='utf-8')
len(kor_corpus)

pos_tagger = Twitter()
stopword = ['것','수','저']
total_news = []
for news in kor_corpus['contents'].head():
    pos_news = ['/'.join(t[:-1]) for t in pos_tagger.pos(news) if ((t[1]=='Noun') & (t[0] not in stopword))]
total_news.append(' '.join(pos_news))

total_news

yonhap = kor_corpus['contents']

import gensim
pip install --upgrade gensim
def hash32(value):
    return hash(value)
doc_list=[]
for doc in yonhap[:1000]:
    tokens = [t for t in pos_tagger.nouns(doc) if len(t)>1]
    doc_list.append(tokens)
    
import time
min_count = 1
hidden_size = 50
workers = 1
window = 3
epoch = 30

start = time.time()
model_total = gensim.models.Word2Vec(doc_list, window = window, min_count=min_count, size=hidden_size, workers = workers, iter = epoch, seed=1, hashfxn=hash32)
end=time.time()
print(end-start)

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument, LabeledSentence

Td = [LabeledSentence(doc,["doc_1"]) for i,doc in enumerate(doc_list)]
d2v = Doc2Vec(Td, window = window, min_count=min_count, size = hidden_size, workers = workers, iter = epoch, seed=1, hashfxn=hash32)

Td

d2v.docvecs.most_similar('doc_1', topn=3)

doc_list
