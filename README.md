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
Boundary Age Group + Boundary Fare Group --> 0.7703 </br>
</br>
Approaches and Results: </br>
| Processed Area in Dataset |  Model  | Result (Accuracy) |
|---------------------------|----------|---------------------|
| Pclass,Age, Sibsp, Parch (all auto-one-hot) | Fully Connected + Dropout | 0.74641|
| Pclass, Age, Sibsp, Parch  (all auto-one-hot) | Fully Connected + Dropout + Batch Norm| 0.73684|
| Pcl, Sibsp, Par (auto-one-hot), Age (binary) | Fully Connected + Dropout | 0.73684|
| Pcl, Sibsp, Par* Age(binary, filled in from name) | Fully Connected + Dropout + Batch Norm | 0.76076|
| Pcl, Sibsp, Par* Age(binary, filled in from name), Fare (binary)| Fully Connected + Dropout + Batch Norm | 0.76794|
| Pcl, Sibsp, Par (one-hot) + Age(boundary group) + Fare (Boundary group) | Fully Connected + Dropout + Batch Norm | 0.7703|
| Pcl, Sibsp, Par (one-hot) + Age(boundary group) + Fare (Boundary group) | RNN | 0.7583|
| Pcl, Sibsp, Par (one-hot) + Age(boundary group) + Fare (Boundary group) | Fully Connected (max. 800 neurons) + Dropout + Batch Norm | 0.7751|


To-do: 
- LeNet-5
- Range for age etc instead of binary (O)
- More precise fare values
