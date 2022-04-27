# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 12:16:31 2020

@author: user
"""

from konlpy.tag import Hannanum
from pprint import pprint
import nltk

import networkx as nx
import matplotlib.pyplot as plt
matplotlib inline
G = nx.MultiGraph()
G.add_nodes_from(['A','B','C','D'])
G.add_edges_from([('A','B'),('A','B'),('A','D'),('B','D'),('B','C'),('B','C'),('C','D')])
nx.draw(G, with_labels=True)
plt.show()

G=nx.Graph()
G.add_edge(1,2)
nx.draw(G, with_labels=True)
plt.show()

#리스트로 추가 1 노드
G = nx.Graph()
G.add_node('apple') 
G.add_nodes_from(['banana','kiwi','mango'])

#리스트로 추가 2 노드
fruits = ['banana','kiwi','mango']
G.add_nodes_from(fruits)

# 노드들 보기
G.nodes() 

nx.draw(G, with_labels=True)
plt.show()

# 리스트로 추가 1 엣지‘apple’과 ‘banana’의 관계(엣지)
G.add_edge('apple','banana') 
G.add_edges_from([('apple','mango'),('apple','kiwi')])
nx.draw(G, with_labels=True)
plt.show()

#리스트로 추가 2 엣지
relations = [('apple','mango'),('apple','kiwi')]
G.add_edges_from(relations) 
G.edges()
nx.draw(G, with_labels=True)
plt.show()

#circular 형태의 그래프
nx.draw_circular(G, with_labels=True)
plt.show()

#spring 형태 그래프
nx.draw_spring(G, with_labels=True)
plt.show()

#spectral 형태 그래프
nx.draw_spectral(G, with_labels=True)
plt.show()

#spectral 레이아웃 점 지정 이전
pos = nx.spectral_layout(G)
print (pos)
nx.draw(G, pos, with_labels=True)
plt.show()

#점 위치 지정 이후
import numpy as np
pos['apple'] = np.array([100,  100])
nx.draw(G, pos, with_labels=True)
plt.show()

G.nodes()

#G에 관한 4x4 행열 생성
nx.to_numpy_matrix(G)

#G Degree계산
G.degree

#G Degree에 따른 node크기 변경 그래프 그리기
degree = nx.degree(G)
degree_dic = dict(degree)

degree = nx.degree(G)
nx.draw(G, node_size=[v*1000 for v in degree_dic.values()], with_labels=True)

#Degree 기준 node sorting
# binomial_graph 생성
G2 = nx.gnp_random_graph(100,0.02)

# Degree 리스트 값 생성 그래프 그리기
degree_sequence=sorted(dict(nx.degree(G2)).values(),reverse=True) 
dmax=max(degree_sequence)
plt.loglog(degree_sequence,'b-',marker='o')
plt.title("Degree rank plot")
plt.ylabel("degree")
plt.xlabel("rank")

#draw graph in inset
plt.axes([0.45,0.45,0.45,0.45])
pos=nx.spring_layout(G2)
plt.axis('off')
nx.draw_networkx_nodes(G2,pos,node_size=20)
nx.draw_networkx_edges(G2,pos,alpha=0.4)
plt.show()

#각 node와 연관된 근접노드/이웃노드 파악
[(n, nbrdict) for n, nbrdict in G.adjacency()]
G.neighbors('apple')
[n for n in G.neighbors('apple')]
[n for n in G.neighbors('banana')]

#가장 많이 연결되어있는 단어 파악
nx.degree_centrality(G)
nx.degree_centrality(doc)
#중요한 흐름을 가진 단어 파악
nx.betweenness_centrality(G)

#접근성이 가장 좋은단어 파악
nx.closeness_centrality(G)

#중요단어에 대한 가중치
nx.eigenvector_centrality(G)


#예시
list1 = ['I', 'love', 'you']
list2 = ['I', 'hate', 'me']
list3 = ['I', 'love', 'me']
list4 = ['I', 'hate', 'boy']
list5 = ['I', 'love', 'you', 'boy']
list = [list1, list2, list3, list4, list5]

G4 = nx.Graph()
for one in list:
    G4.add_nodes_from(one)
    print(one)
for j in range(1,len(one)-1):
    print('j=', j)
for i in range(len(one)-j):
    print('i=', i)
G4.add_edge(one[i], one[i+j])

nx.draw(G4, with_labels=True)
nx.draw_shell(G4, with_labels=True)
nx.draw_kamada_kawai(G4, with_labels=True)
nx.draw_circular(G4, with_labels=True)
nx.draw_spring(G4, with_labels=True)
pos = nx.spectral_layout(G)

nx.betweenness_centrality(G4)
nx.closeness_centrality(G4)
nx.degree_centrality(G4)


import os
os.chdir("C:/Users/user/Desktop/PYJ/비정형/3주차")
doc = open('test.txt','r', encoding='utf-8').read()
doc
copus = doc.split('.')
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(1,1), stop_words = 'english') # You can define your own parameters\n",
X = cv.fit_transform(copus)

Co_ocurrence = (X.T * X) # This is the matrix manipulation step
Co_ocurrence.setdiag(0)
Co_ocurrence
nx.degree_centrality(X)

import pandas as pd
names = cv.get_feature_names() # This are the entity names (i.e. keywords)
df = pd.DataFrame(data = Co_ocurrence.toarray(), columns = names, index = names)
df.to_csv('to gephi.csv', sep = ',')

df

per

