---
title : Activation Function
author: MinGiSa
date : 2023-02-14 17:30:00 +0900
categories : [DeepLearning]
tags : [deeplearning, activation function]
---
## Activation Function
---
<br>
<br>
## Activation Function / 활성화 함수
---
<br>
- 입력 값을 바로 다음 레이어로 전달하지 않고 주로 비선형 함수를 통과시켜 전달
<br>
- 입력 신호의 총합을 출력 신호로 변환하는 함수
<br>
- Transfer Function(전달함수)이라고 부르기도 함
<br>
- Activation Function을 사용한다는 것은 입력을 Normalization(일반화)을 하는 것으로 볼 수 있음
<br>
- Bias(편향) +(Node(입력값 또는 출력값) * Weight(가중치)) = 활성화 여부 결정
<br>
- 각 노드에서 연산된 값을 특정 Threshold(임계값) 기준으로 다음 레이어 전달 여부를 정함
<br>
-> 스위치 On / Off 역할과 유사함
<br>
-> 약한 특징을 제외하고 강한 특징들만 추려내기 유리
<br>
- 선형 함수 사용시 신경망의 Hidden Layer(은닉층) 구성 의미가 줄어듬
<br>
-> f(x)=Yx 에서 Hidden Layer 2개 추가해도 항상 상수 출력WW
<br>
-> y(x)f(f(f(x))) = W^3x
<br>
-> 하나의 레이어로 다층 레이어 표현 가능 (Hidden Layer가 없는 것과 같음)
<br>
-> 또한, Weight가 갱신되지 않고 항상 같은 값을 반환하기에 역전파 및 다중 출력 불가능
<br>
<br>

## 1. Step Function / 계단 함수
---
- Perceptron(퍼셉트론)에서 처음으로 사용한 Activation Function
<br>
- XOR GATE 진리값 그래프 하나의 선으로 구분 불가능, 이를 해결하기 위하여 MLP 사용
<br>
- 입력 값의 합이 Threshold를 넘기면 0, 넘지 못하면 0 을 출력
<br>
- x < 0 = 0 / - x >= 0 = 1
<br>
- Threshold(x = 0) 구간에서 Point of Discontinuity(불연속점)을 가지므로 미분 불가능
<br>
- Threshold(x >= 0, x < 0) 구간에서 미분 시 모두 0 값으로 변경되어 Back Propagation(역전파) 과정에서 학습 불가능
<br>
- 따라서, Deep Learning 모델에서는 사용되지 않음
<br>
<br>

## 2. Sigmoid Function / 시그모이드 함수
---
- Step Function 보완
<br>
->전구간 미분 가능 / Gradient 값 : 0 ~ 0.25
<br>
- 실수 값을 입력 받아 0과 1 사이로 출력 값 압축 / 중심 0.5(not zero-centered)
<br>
- 입력값이 커질수록 출력값은 1에 수렴 / 작아질수록 0에 수렴
<br>
- 입력값이 커지고 작아짐에 따라 Gradient(기울기)는 0에 수렴
<br>
- Binary Classification(이진분류)에서 주로 사용
<br>
- Vanishing Gradient(기울기 소실) 문제 발생으로 학습 성능 저하
<br>
-> 모델의 훈련이 느려지거나 안되는 문제
<br>
-> Back Propagation 진행 중에 이전의 Gradient와 현재의 Gradient를 곱하면서 점점 Gradient가 사라지고 Saturation(포화현상) 발생
<br>
- 출력 값의 중심이 0이 아니기 때문에 학습 속도가 느려짐
<br>
-> Gradient 값이 모두 양수 또는 음수만 존재
-> Gradient Descent(경사하강법)으로 Weight 업데이트시 대각선 방향이 한쪽으로 편향이동 발생(지그재그 현상)으로 최적화에 어려움 발생

- exp(지수함수) 연산으로 Cost(비용) 소모가 큼

## 3. Tanh Function(Hyperbolic Tangent) / 탄젠트 함수
---
<br>
- Sigmoid Function 보완
-> 실수값을 입력 받아 -1과 1 사이로 출력 값 압축 / 중심 0(zero-centered)
<br>
-> 상대적으로 최적화 성능 우수
<br>
- 입력값이 커질수록 출력값은 1에 수렴 / 작아질수록 0에 수렴
<br>
- 입력값이 커지고 작아짐에 따라 Gradient는 0에 수렴
<br>
- Vanishing Gradient(기울기 소실) 문제 발생으로 학습 성능 저하
<br>
-> 모델의 훈련이 느려지거나 안되는 문제
<br>
-> Back Propagation 진행 중에 이전의 Gradient와 현재의 Gradient를 곱하면서 점점 Gradient가 사라지고 Saturation(포화현상) 발생
<br>
<br>

## 4. ReLU Function(Recified Linear Unit) / 렐루 함수
---
<br>
- Sigmod / Tanh Function 보완
<br>
-> 입력값이 양수면 출력값은 입력값과 같음 / 음수면 출력값은 0
<br>
-> Gradient 값이 1이기에 Vanishing Gradient 문제 해결
<br>
   -> 지그재그 현상은 발생함에 따라 
<br>
-> exp 연산이 없어 상대적으로 계산 속도 우수(미분값 0 또는 1)
<br>
- Dying ReLu 문제 발생으로 학습 성능 저하
<br>
-> 중심값이 0이 아니며 x가 음수면 출력값이 0이기에 일부 Weight 갱신 불가
<br>
-> Batch Norm 으로 입력값을 0 근처의 분포로 강제하여 해결 가능
<br>
-> Sparse Representation(희소 표현성)이 부여됨에 따라 특정 상황에서는 문제가 되지 않음
<br>
   -> Deep & Wide한 신경망에서는 오히려 Drop-Out과 비슷한 역할을 함
<br>
   -> 일부 Node가 죽음에 따라 Drop-Out처럼 네트워크 상관성 감소로 overfitting 방지
<br>
<br>

5. SoftMax Function / 소프트맥스 함수
<br>
- 입력값을 세개 이상으로 분류하는 Muili Classification(다중 클래스) 분류 모델에서 사용
<br>
-> n개의 클래스를 n차원의 벡터로 입력 받아 각 Class에 속할 확률 추정 / 출력값의 총 합은 1
<br>
- 실수값을 입력받아 0과 1사이로 출력 값 압축(정규화)
<br>
- 특정 조건에서 큰 입력값은 강조, 작은 입력값은 억제
<br>
지수함수 사용으로 OverFlow 발생 가능
<br>
-> 지수함수가 단조 증가 함수로 대소관계 미변화
<br>
-> 입력값의 최댓값을 제하는걸로 방지
<br>
<br>
<br>
<br>
<br>
출처 :
<br> 
https://brunch.co.kr/@kdh7575070/27
<br>
https://happy-obok.tistory.com/55
<br>
https://mole-starseeker.tistory.com/39
<br>
https://cheris8.github.io/artificial%20intelligence/DL-Activation-Function/
<br>
https://wooono.tistory.com/209
<br>
https://wikidocs.net/60683
<br>
https://velog.io/@joo4438/%EB%94%A5%EB%9F%AC%EB%8B%9D-%ED%99%9C%EC%84%B1%ED%95%A8%EC%88%98-Activation-function
<br>
https://076923.github.io/posts/AI-6/
<br>
https://gooopy.tistory.com/54
<br>
https://soki.tistory.com/63
<br>
https://gooopy.tistory.com/53
<br>
https://076923.github.io/posts/AI-6/
<br>
https://velog.io/@gjtang/Softmax-%ED%95%A8%EC%88%98%EB%9E%80
<br>