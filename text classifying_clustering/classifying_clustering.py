# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 23:57:33 2020

@author: young
"""
##########################classifying###############################
pip install keras 
pip install tesorflow -gpu
tensorflow.__version__
pip install --upgrade tensorflow
pip install --upgrade pip

import tensorflow
import keras
keras.__version__
from keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
len(train_data)
len(test_data)

train_data[10]

word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# 0, 1, 2는 '패딩', '문서 시작', '사전에 없음'을 위한 인덱스이므로 3을 뺍니다
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
decoded_newswire

train_labels[10]

import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_train
x_test = vectorize_sequences(test_data)
x_test

from keras.utils.np_utils import to_categorical

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # 그래프를 초기화합니다

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train,
          partial_y_train,
          epochs=9,
          batch_size=512,
          validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)
results

###########clustering###################
import konlpy
from konlpy.tag import Hannanum
from pprint import pprint
import nltk

import os
import pandas as pd
from konlpy.tag import Twitter

kor_corpus = pd.read_csv('news_koreanwave.csv', encoding='utf-8')
kor_corpus.head()

pos_tagger = Twitter()

stopword = ['것','수','저']
total_news = []
for news in kor_corpus['contents'].head():
    pos_news = ['/'.join(t[:-1]) for t in pos_tagger.pos(news) if ((t[1]=='Noun') & (t[0] not in stopword))]
    total_news.append(' '.join(pos_news))
total_news


from sklearn.feature_extraction.text import CountVectorizer

kor_vectorizer = CountVectorizer(min_df=1) # 등장하는 단어들에 대한 오브젝트
kor_bow = kor_vectorizer.fit_transform(total_news) # 딕셔너리에 실제 단어들을 입력
print(kor_bow)

print(kor_vectorizer.get_feature_names())
# bag-of-words
print(kor_bow.shape)
print(kor_bow.toarray())

from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer() # tfidf 변환 인스턴스 생성
tfidf = transformer.fit_transform(kor_bow.toarray())
tfidf.toarray()


yonhap = kor_corpus['contents']

import re

total_news = []
stopword = ['것','수','저']

for news in yonhap[:1000]:
    pos_news = re.sub('.* 기자 = ', '', news) # 한 기자가 많은 기사를 작성. 기자명 때문에 유사한 것으로 계산될 수 있으니 이를 제거하자
    pos_news = ['/'.join(t[:-1]) for t in pos_tagger.pos(pos_news) if ((t[1]=='Noun') & (t[0] not in stopword))]
    total_news.append(' '.join(pos_news))
kor_vectorizer = CountVectorizer(min_df=10)
kor_bow = kor_vectorizer.fit_transform(total_news)
    
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(kor_bow[0], kor_bow)
(-cosine_similarity(kor_bow[0], kor_bow)).argsort()
idx = (-cosine_similarity(kor_bow[0], kor_bow)[0]).argsort()
print(idx)


def find_most_similar_news(index, bow, corpus):
    idx = (-cosine_similarity(bow[index], bow)[0]).argsort()[1]
    return corpus[idx]

idx = 300
yonhap[idx]
find_most_similar_news(idx, kor_bow, yonhap) # Query를 제외하고 가장 유사한 문서, 북한이라는 키워드는 없다

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(kor_bow.toarray())

def find_most_similar_news_idf(index, tfidf, corpus):
    idx = (-cosine_similarity(tfidf[index], tfidf)[0]).argsort()[1]
    return corpus[idx]

yonhap[idx]
find_most_similar_news_idf(idx, tfidf, yonhap)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, n_init=50, verbose=1)
kmeans.fit(tfidf)
kmeans.labels_

import numpy as np

clusters = []

for i in range(0, kmeans.n_clusters):
    clusters.append(np.where(kmeans.labels_ == i))
clusters

print('1번 클러스터')
print(yonhap[8])
print(yonhap[9])
print(yonhap[19])

print('2번 클러스터')
print(yonhap[35])
print("\n")
print(yonhap[36])
print("\n")
print(yonhap[37])
print(yonhap[38])

import gensim


#아래 명령어 실행
def hash32(value):
     return hash(value) & 0xffffffff
    
doc_list = []
for doc in yonhap[:1000]:
    tokens = [t for t in pos_tagger.nouns(doc) if len(t)>1]
    doc_list.append(tokens)
    
import time
min_count = 1 #2 번 미만으로 나온 단어는 무시하겠다
hidden_size = 50 #corpus 가 적을때는 50, 30 정도만 해도 충분하다, 많으면 100~200
workers = 1 #core 수
window= 3 #corpus가 작으면 window size가 적은게 좋고, 많으면 5~10으로 해도 됨
epoch = 30  #30~50정도

start = time.time()
model_total = gensim.models.Word2Vec(doc_list, window = window, min_count=min_count, size = hidden_size, workers = workers, iter = epoch, seed=1, hashfxn=hash32)

end = time.time()
print(end-start)
model_total.most_similar('드라마')
model_total
doc_list
