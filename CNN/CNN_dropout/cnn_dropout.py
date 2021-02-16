import tensorflow as tf
import numpy as np
import ast
from tensorflow.keras import layers, models

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


x, y = open('train_fare.txt','r',encoding = 'utf8').readlines(), open('train_label.txt','r',encoding ='utf8').readlines()
x, y = [ast.literal_eval(i) for i in x], [ast.literal_eval(j) for j in y]
train_x1 = np.array(x)
train_x = np.array([[train_x1[i]]*len(train_x1[i]) for i in range(len(train_x1))])
train_x2 = train_x.reshape((-1,47,47,1))
train_y2 = np.array([i[1] for i in y])

train_x,train_y = np.array(train_x2), np.array(train_y2)
#train_x,train_y = np.array(train_x2[:802]), np.array(train_y2[:802])
val_x, val_y = np.array(train_x2[802:]), np.array(train_y2[802:])


test_x = open('test_age_fare.txt','r',encoding = 'utf8').readlines()
test_x = [ast.literal_eval(j) for j in test_x]
text_x1 = np.array(test_x)
test_x = np.array([[text_x1[i]]*len(text_x1[i]) for i in range(len(text_x1))])
test_x = test_x.reshape((-1,47,47,1))



model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation= 'relu', input_shape = (47,47,1)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Dropout(0.5))

model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.5))

model.add(layers.Conv2D(64,(3,3), activation ='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(32,activation = 'relu'))
model.add(layers.Dense(2, activation = 'softmax'))
model.compile(optimizer = 'adam',loss= 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(train_x, train_y, epochs = 5000)
val_loss, val_acc = model.evaluate(val_x, val_y, verbose = 2)
train_outputs = []
outputs = []
for i in range(len(test_x)):
    outputs.append(int(tf.argmax(model.predict(test_x[i:i+1]),1)))
for j in range(len(val_x)):
    train_outputs.append(int(tf.argmax(model.predict(val_x[j:j+1]),1)))
print(val_acc)

import csv
list1 = [892+i for i in range(len(test_x))]
list2 = outputs

d = zip(list1, list2)
with open('output_cnn.csv', 'w',encoding = 'utf8') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(("PassengerId", "Survived"))
    wr.writerows(d)
myfile.close()
