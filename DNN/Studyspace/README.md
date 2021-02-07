Realizing the importance of data processing: </br>
주어지는 학습 데이터 중 비어있는 칸이나 one hot으로 변환하는 부분에 대해 적절히 처리를 하면 정확도가 향상되는 결과 볼 수 있었음. </br>
</br>
Hyperparameter 튜닝: </br>
1. Learning Rate </br>
2. Batch size </br>
</br>
각각에 대해 validation set 중 가장 좋은 결과를 내는 수치를 측정해 최고의 성능을 보이는 파라미터 값이 lr = 0.95, batch size = 19 로 나왔다. </br>
그럼에도 기존 모델보다 안좋은 결과를 볼 수 있었고, 추가적인 데이터 처리 + 모델 수정에 대한 연구 필요. 
