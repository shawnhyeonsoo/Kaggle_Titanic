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

|Model| Layers | Accuracy (Kaggle_score)|
|-----|--------|---------|
|CNN | 3 conv2d, 2 max pooling, 3 Fully Connected | 0.622|
|CNN (5000 epoch)| 3 conv2d, 2 max pooling, 3 Fully Connected| 0.722|
|CNN (5000 epoch + Dropout)| 3 conv2d, 2 max pooling, 3 Fully Connected + 3 Dropout |0.715|


Epoch를 늘림으로써 오버피팅의 여파가 없지않아 보인다. Validation set으로 테스트하는 성능에서는 낮은 epoch 값에서도 90퍼센트 가까이로 높은 정확도를 보이는 반면, keras에서 측정하는 정확도는 공개되지 않은 테스트 레이블에 대해 측정함으로, 데이터를 처리하는 과정의 중요성이 더 크게 느껴진다. 
