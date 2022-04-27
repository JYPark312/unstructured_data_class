# 비정형 데이터 실습

## 간단 신문 기사 크롤링 후 pos tagger, tokenizing
- 한글의 경우 교착어이기 때문에 토큰화가 어렵다
- 토큰화를 할 경우 형태소 단위로 분리하여 토큰화를 진행해야한다
- 한글은 띄어쓰기가 영어보다 잘 지켜지지 않는다 
- 이러한 특성으로 인해 한국어의 경우 nlp모델에 따라 pos tagger, tokenizing 결과가 달라진다

### crawling result
![image](https://user-images.githubusercontent.com/70933580/165443349-288843f3-d2d3-43ef-9ca3-9137b3794a1d.png)

### 꼬꼬마 pos tagger
![image](https://user-images.githubusercontent.com/70933580/165443390-ea77ed56-a92a-40dd-b19f-4dc3cb24bac3.png)

### twitter pos tagger
![image](https://user-images.githubusercontent.com/70933580/165443432-1649172d-0051-4b4a-9aeb-ff9971335f70.png)

### 한나눔 pos tagger
![image](https://user-images.githubusercontent.com/70933580/165443939-4b9cf2a0-fc3b-4248-ae07-7e07fa12d99b.png)

- 꼬꼬마는 한나눔에 비해 세부적으로 태깅을 함
- 그러나 지명, 사람이름 등 한가지로 분류되어야 할 명사가 나눠지는 경우들이 생김
- 트위터는 사람 이름을 정확하게 분류하지 못하는 경우가 생김
- 한나눔이 비교적 사람의 이름이나 지면 등을 잘 구분해냄

### 꼬꼬마 tokenizing
![image](https://user-images.githubusercontent.com/70933580/165444217-ced259c8-44ac-41f3-be96-c13bd9278007.png)

### twitter tokenizing
![image](https://user-images.githubusercontent.com/70933580/165444204-bac689a2-0483-4210-a797-afc66f285d53.png)

### 한나눔 tokenizing
![image](https://user-images.githubusercontent.com/70933580/165444187-41c35815-f44b-462b-971c-96a178bdd205.png)

- pos tagging을 할 때 tokenizing이 같이 이루어지기 때문에 결과가 비슷함

### 꼬꼬마 명사 추출
![image](https://user-images.githubusercontent.com/70933580/165444330-39473f6c-9466-4a03-bb31-be7d67450191.png)

### twitter 명사 추출
![image](https://user-images.githubusercontent.com/70933580/165444345-9d293f3e-8e76-4fc3-ba1d-46a763d08cba.png)

### 한나눔 명사 추출
![image](https://user-images.githubusercontent.com/70933580/165444359-b60e535c-c3db-4f24-b53c-a7ab364312ec.png)

- 명사 분류 능력은 한나눔과 트위터가 비슷, 꼬꼬마는 너무 잘게 쪼개는 특성을 
