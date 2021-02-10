# Kaggle_Titanic
Basic Kaggle ML Prediction competition challenge- Titanic Survival Prediction
</br>
https://www.kaggle.com/c/titanic
</br>
</br>


0.74641</br>
Binary Age --> 0.76076
</br>
Binary Age + Binary Fare --> 0.76794</br>
</br>
Approaches and Results: </br>
| Processed Area in Dataset |  Model  | Result (Accuracy) |
|---------------------------|----------|---------------------|
| Pclass,Age, Sibsp, Parch (all auto-one-hot) | Fully Connected + Dropout | 0.74641|
| Pclass, Age, Sibsp, Parch  (all auto-one-hot) | Fully Connected + Dropout + Batch Norm| 0.73684|
| Pcl, Sibsp, Par (auto-one-hot), Age (binary) | Fully Connected + Dropout | 0.73684|
| Pcl, Sibsp, Par* Age(binary, filled in from name) | Fully Connected + Dropout + Batch Norm | 0.76076|
| Pcl, Sibsp, Par* Age(binary, filled in from name), Fare (binary)| Fully Connected + Dropout + Batch Norm | 0.76794|


To-do: 
- CNN
- LeNet-5
