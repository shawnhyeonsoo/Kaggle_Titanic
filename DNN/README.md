"First Participation" </br></br>
This was my first participation in a prediction challenge. </br>
Before getting deep into all the possible research for accuracy enhancement, I tried to get familiar with such challenges.</br>
This folder contains the basic 'Fully Connected Layers' models: one with Dropout, and another with Dropout + Batch Normalization. </br>
The Dropout model scored 0.74641 in accuracy, and the second scored 0.73684. Possibly missed out something along the way. To be updated. </br>
</br>

처음으로 참가하게 된 캐글 예측 모델 챌린지였다. </br>
제대로 들어가기에 앞서, 우선 전체적인 챌린지의 흐름을 파악하고자 간단한 Fully Connected Layer 모델을 구현해 보았다. </br>
본 폴더에는 Fully Connected Layers + dropout 모델과 여기에 Batch Normalization까지 추가한 모델의 코드가 있다. </br>
결과로는 드랍아웃 모델이 75% 정도의 정확도를 보였고, Batch Normalization까지 추가한 모델이 조금 낮은 수치인 74% 정도의 수치를 보였다. 구현함에 있어 놓친 부분도 있을 것이라 생각된다. 
</br>
</br>
Approaches and Results: </br>
| Processed Area in Dataset |  Model  | Result (Accuracy) |
|---------------------------|----------|---------------------|
| Pclass,Age, Sibsp, Parch (all auto-one-hot) | Fully Connected + Dropout | 0.74641|
| Pclass, Age, Sibsp, Parch  (all auto-one-hot) | Fully Connected + Dropout + Batch Norm| 0.73684|
| Pcl, Sibsp, Par (auto-one-hot), Age (binary) | Fully Connected + Dropout | 0.73684|
| Pcl, Sibsp, Par* Age(binary, filled in from name) | Fully Connected + Dropout + Batch Norm | 0.76076|
| Pcl, Sibsp, Par* Age(binary, filled in from name), Fare (binary)| Fully Connected + Dropout + Batch Norm | 0.76794|

</br>
</br>
Hyperparams:

| Learning Rate | Batch Size|
| -------------|--------|
|0.05||


</br>
> 2021.03.10 (Wed) </br>
- Updated DNN parameters
- Weights (39,400,800,2) 일때 0.7751로 최고 정확도
