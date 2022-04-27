# 비정형 데이터 

## Word2Vec, LSA

- word2vec: class softmax regression을 이용하여 각 단어의 벡터를 학습시키는 classifier
  - 기본개념
  - softmax regression은 한 단어의 벡터를 학습하기 위해 v개의 모든 단어의 벡터를 수정하기 때문에 계산량이 매우 큼
  - 단어간의 관계성이 추출되는 효과가 있음
  - CBow: 주위 단어로 현재 단어를 예측하는 모델 / Skipgram 현재 단어로 주위 단어를 모두 예측하는 모델
![image](https://user-images.githubusercontent.com/70933580/165459094-6cf61360-e51b-46e0-968d-3499303a3eb8.png)

- LSA(잠재 의미 분석) 
  - BoW에 기반한 DTM이나 TF-IDF는 기본적으로 단어의 빈도 수를 이용한 수치화 방법이기 때문에 단어의 의미를 고려하지 못함
  - 위의 문제를 해결하기 위해 DTM에 SVD(singular value decomposition)을 적용한 LSA가 개발 
  - SVD: A가 m x n 행렬일 때, 다음 아래의 3개 행렬의 곱으로 분해되는 기법
  
  ![image](https://user-images.githubusercontent.com/70933580/165463118-4392b29e-1e09-48e7-8b30-bde49882d6bc.png)

  ![image](https://user-images.githubusercontent.com/70933580/165463142-d04c8635-0b50-44d9-a060-e828cf1a5619.png)
