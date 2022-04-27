# 비정형 데이터 

## TEXT NETWORK

### text file 의 word network 구현
- text file을 읽어옴
- ngram 1, stop_words english 로 counter vectorize 설정
- text file을 counter vectorize transform 시킴
- coocurrence matrix 생성(x와 x.T를 곱해서)

### coocurrence matrix 결과
![image](https://user-images.githubusercontent.com/70933580/165447118-a7e8191b-1b65-441d-8cae-aefcaadc2853.png)

### Gephi 시각화
- degree 2-10

![image](https://user-images.githubusercontent.com/70933580/165447212-38f1735e-b775-45d2-8cf4-75ff6697c957.png)

- degree 2-20

![image](https://user-images.githubusercontent.com/70933580/165447250-e2e3b746-cfce-4348-a6ec-0b8c325fade8.png)

- degree 2-30

![image](https://user-images.githubusercontent.com/70933580/165447294-a9eb2f72-8d31-46ec-8a43-44b3a9326b67.png)

- 초기 gephi 사용법 미숙으로 node명 표시 하지 못한 한계점
- 추후 연구들에서는 node의 명을 표시하였음
