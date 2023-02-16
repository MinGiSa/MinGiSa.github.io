---
title : Normalization, Standardization, Regularization
author: MinGiSa
date : 2023-02-15 17:30:00 +0900
categories : [DeepLearning]
tags : [deeplearning, Normalization, Standardization, Regularization]
---
## Normalization, Standardization, Regularization
---
<br>
<br>
## ## Normalization, Standardization, Regularization / 일반화, 표준화, 정규화
---
<br>
- 데이터 전처리시 Noise Data 생성 및 OverFitting 발생 방지를 주 목적으로 사용하는 기법 
<br>
- Feature Scaling(특성 스케일링)이라고 부르기도 함
<br>
- 해당 기법 사용 여부 및 순서는 데이터 특성에 따라 유동적이나 개인적으로 Normalization -> Standardization -> Regularization 진행
<br>
- 각 용어의 사용이 혼재되어 있는 상황
<br>
-> Normalization, Regularization을 정규화 / Normalization, Standardization, Regularization 전부 정규화로 부르기도 하기에 한글 사용 지양
<br>
<br>
<br>
## 1. Normalization / 일반화
---
<br>
- Scale이 큰 입력값의 영향 증가 방지
<br>
- 값을 입력 받아 0과 1 사이로 출력 값 압축
<br>
- 값의 범위가 다른 경우에 사용
<br>
- 값의 규모 축소 (Feature Scaling)
<br>
-> Cost 감소로 학습 속도 증가(Local Minima(국소값) 문제 발생 가능성이 낮아짐)
<br>
- Min / Max 편차가 크거나 지나치게 큰 값에 사용
<br>
- 노이즈 생성 방지
<br>
- OverFitting 방지
<br>
- 일반적으로 Batch Norm / MinMaxScaler 사용
<br>
1-1. Batch Norm
<br>
- 기존 Whitening 방식은 계산량이 많고 일부 Bias(파라미터)들의 영향이 무시
<br>
- 일반적으로 Fully-connected layer 뒤,  non-linearity function 앞에서 실행
<br>
1-2. Min / Max
<br>
- 데이터 최소값, 최대값 제외
<br>
<br>
## 2. Standardization / 표준화
---
<br>
- 데이터 분포를 정규분포로 변환 / 데이터 평균 : 0, 표준 편차 : 1 (Zero-Centered)
<br>
- OutLier(이상치) 제거 및 노이지 생성 방지
<br>
- 값을 입력 받아 출력 값을 압축하지만 해당 범위는 특정 범위로 제한하지 않음
<br>
- 일반적으로 z-score 사용
<br>
-> (값-평균)/표준편차
<br>
- 값의 규모 축소 (feature scaling)
<br>
-> Cost 감소로 학습 속도 증가(Local Minima(국소값) 문제 발생 가능성이 낮아짐)
<br>
- OverFitting 방지
<br>
<br>
## 3. Regularization / 정규화
---
<br>
- Weight 제약 발동
<br>
- OverFitting 방지
<br>
- 값의 규모 축소 (Feature Scaling)
<br>
-> Cost 감소로 학습 속도 증가(Local Minima(국소값) 문제 발생 가능성이 낮아짐)
<br>
- 일반적으로 L1 / L2 Regularization / Drop-Out 사용
<br>
<br>
3-1. L1 Norm(Mahattan Distance, Taxicab geometry)
<br>
- 특정 변수가 미치는 영향을 “제어”하겠다는 접근 방식 / 변수 선택 가능
<br>
- 변수간 상관관계가 높으면 성능 감소
<br>
- 비중요 변수를 우선적으로 줄임
<br>
<br>
3-2. L2 Norm(Euclidean Distance)
<br>
- 계산에 들어간 모든 변수의 영향을 “제어”하겠다는 접근 방식 / 변수 선택 불가능
<br>
- 변수간 상관관계가 높아도 성능 증가
<br>
- 크기가 큰 변수를 우선적으로 줄임
<br>
<br>
3-3. Elasticnet
<br>
- 변수간 상관관계를 반영 / 변수 선택 가능
<br>
- 상관관계가 큰 변수를 동시에 선택 및 배제
<br>
<br>
<br>
<br>
<br>
출처 :
<br>
https://moojuksublime.tistory.com/18
<br>
https://jaeyung1001.tistory.com/169
<br>
https://pdsi.pabii.com/l1-l2-regularization-intuitive-understanding/
<br>
https://huidea.tistory.com/154