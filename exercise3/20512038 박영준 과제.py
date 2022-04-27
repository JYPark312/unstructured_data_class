# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 16:08:22 2020

@author: user
"""

from konlpy.tag import Hannanum
from pprint import pprint
import nltk
pos_tagger = Hannanum()
from sklearn.feature_extraction.text import CountVectorizer
from konlpy.tag import Twitter
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from konlpy.utils import pprint
from nltk import collocations
from konlpy.tag import Kkma
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfTransformer
import os
os.chdir()
kor_corpus = pd.read_csv('C:/Users/user/Desktop/PYJ/비정형/5주차/news_koreanwave.csv', encoding='utf-8')
kor_corpus.head()

len(kor_corpus)
pos_tagger= Twitter()

stopword = ['것','수','저']
    total_news = []
    for news in kor_corpus['contents'].head():
        pos_news = ['/'.join(t[:-1]) for t in pos_tagger.pos(news) if ((t[1]=='Noun') & (t[0] not in stopword))]
        total_news.append(' '.join(pos_news))
        
total_news
kor_vectorizer = CountVectorizer(min_df=1) # 등장하는 단어들에 대한 오브젝트
kor_bow = kor_vectorizer.fit_transform(total_news) # 딕셔너리에 실제 단어들을 입력
print(kor_bow)

print(kor_vectorizer.get_feature_names())
# bag-of-words
print(kor_bow.shape)
print(kor_bow.toarray())

transformer = TfidfTransformer() # tfidf 변환 인스턴스 생성
tfidf = transformer.fit_transform(kor_bow.toarray())

tfidf.toarray()

#연어 
from konlpy.utils import pprint
from nltk import collocations
from konlpy.tag import Kkma
 
measures = collocations.BigramAssocMeasures()
doc = open('C:/Users/user/Desktop/PYJ/비정형/2주차/final_announcement_for_Park.txt', 'r', encoding='utf-8').read()
print('Collocations among tagged words:')
tagged_words = Kkma().pos(doc)
finder = collocations.BigramCollocationFinder.from_words(tagged_words)
pprint(finder.nbest(measures.pmi, 10)) # top 10 n-grams with highest PMI

print('Collocations among words:')
words = [w for w, t in tagged_words]
ignored_words = [u'안녕']
finder = collocations.BigramCollocationFinder.from_words(words)

finder.apply_word_filter(lambda w: len(w) < 2 or w in ignored_words)
finder.apply_freq_filter(3) # only bigrams that appear 3+ times
pprint(finder.nbest(measures.pmi, 10))

print('Collocations among tags:')
tags = [t for w, t in tagged_words]
finder = collocations.BigramCollocationFinder.from_words(tags)
pprint(finder.nbest(measures.pmi, 5))

#유사문서 검색
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfTransformer

wordcloud = WordCloud(font_path = \"/Library/Fonts/AppleGothic.ttf\")
                      
#SVD분해 
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
X = sparse_random_matrix(100, 100, density=0.01, random_state=42)
svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
svd.fit(X)  
TruncatedSVD(algorithm='randomized', n_components=5, n_iter=7,
            random_state=42, tol=0.0)
print(svd.explained_variance_ratio_)
print(svd.explained_variance_ratio_.sum())
print(svd.singular_values_)

#LSA모델
from gensim.test.utils import common_dictionary, common_corpus
from gensim.models import LsiModel

model = LsiModel(common_corpus, id2word=common_dictionary)
vectorized_corpus = model[common_corpus]

vectorized_corpus.corpus