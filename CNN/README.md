# Implementation/ Testing with CNN model
CNN Model: </br>

- c1: Convolution, 32 filters with (3x3), activation = 'relu'
- s2: Maxpooling, (2x2)
- c3 : Convolution, 64 filters with (3x3), activation = 'relu'
- s4 : Maxpooling, (2x2)
- c5 : Convolution, 64 filters with (3x3), activation = 'relu'
- f6: 64 neurons, activation = 'relu'
- f7: 32 neurons, activation = 'relu'
- output: 2 neurons, activation = 'softmax'

기존 전처리한 학습 데이터가 (891 x 47) 형식으로 되어 있을 때, 각 데이터를 정사각형 데이터를 위해 복제를 진행, 전체적인 학습 데이터를 (891 x 47 x 47 x 1) 형식으로 변형시킨 후,
모델을 학습하는데에 사용했다. 

모델의 성능을 확인하기 위해 학습 데이터의 10%를 Validation 데이터 겸 모델 성능 테스트용으로 분리해 실험을 진행했다.
모델이 정상적으로 동작함을 확인하는 1차 실험에서의 정확도는 76.4% 
