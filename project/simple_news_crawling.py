# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 12:12:49 2020

@author: user
"""

pip install konlpy
import os
os.chdir('C:/Users/user/Desktop/PYJ/비정형/2주차')
doc = open('Son_on_fire.txt','r', encoding='utf-8').read()
from konlpy.tag import Kkma, Twitter, Hannanum
kkma=Kkma()
twitter=Twitter()
hannanum=Hannanum()
kkma.pos(doc)
twitter.pos(doc)
hannanum.pos(doc)
from pprint import pprint


pos_taggers = [('kkma', Kkma()), ('twitter', Twitter()), ('hannanum', Hannanum())]
for name, tagger in pos_taggers:
    print('tagger name = %10s\\tclass_name = %s' % (name, tagger.__class__))

for name, tagger in pos_taggers:
    output = pos_taggers(doc)
    print('==============================')
    print('tagger name= ', name)
    pprint(output)

doc
doc = doc.replace('\n', ' ')
doc
    
#꼬꼬마 트위터 한나눔으로 pos태깅 작업
kkma.pos(doc)
twitter.pos(doc)
hannanum.pos(doc)
pos_taggers = [('kkma', Kkma()), ('twitter', Twitter()), ('hannanum', Hannanum())]

#토크나이징 작업
tokens = []
import time

for name, tagger in pos_taggers:
    _tokens = tagger.pos(doc)
    tokens.append(_tokens)

len(tokens)
tokens[0]
tokens[1]
tokens[2]


from collections import Counter
#꼬꼬마 명사추출
counter = Counter(tokens[0])
counter = {word:freq for word, freq in counter.items() if (freq >= 2) and (word[1][:2] == 'NN')}
pprint(sorted(counter.items(), key=lambda x:x[1], reverse=True))

#트위터 명사추출
counter = Counter(tokens[1])
counter = {word:freq for word, freq in counter.items() if (freq >= 2) and (word[1][:2] == 'No')}
pprint(sorted(counter.items(), key=lambda x:x[1], reverse=True))

#한나눔 명사추출
counter = Counter(tokens[2])
counter = {word:freq for word, freq in counter.items() if (freq >= 2) and (word[1][:1] == 'N')}
pprint(sorted(counter.items(), key=lambda x:x[1], reverse=True))
