# 비정형 데이터 

## Word2Vec, LSA

- word2vec: class softmax regression을 이용하여 각 단어의 벡터를 학습시키는 classifier
  - 기본개념
![image](https://user-images.githubusercontent.com/70933580/165459094-6cf61360-e51b-46e0-968d-3499303a3eb8.png)
  - softmax regression은 한 단어의 벡터를 학습하기 위해 v개의 모든 단어의 벡터를 수정하기 때문에 계산량이 매우 큼
  - 단어간의 관계성이 추출되는 효과가 있음
  - CBow: 주위 단어로 현재 단어를 예측하는 모델 / Skipgram 현재 단어로 주위 단어를 모두 예측하는 모델

- LSA(잠재 의미 분석) 
 
