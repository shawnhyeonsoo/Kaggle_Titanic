import tensorflow.compat.v1 as tf
import numpy as np
import ast

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.disable_eager_execution()
tf.disable_v2_behavior()

x, y = open('train_boundaries.txt','r',encoding = 'utf8').readlines(), open('train_label.txt','r',encoding ='utf8').readlines()
x, y = [ast.literal_eval(i) for i in x], [ast.literal_eval(j) for j in y]
train_x,train_y = np.array(x), np.array(y)
train_x = train_x.reshape(891,39,1)
#train_x,train_y = np.array(x[:802]), np.array(y[:802])
val_x, val_y = np.array(x[802:]), np.array(y[802:])
val_x = val_x.reshape(89,39,1)

test_x = open('test_boundaries.txt','r',encoding = 'utf8').readlines()
test_x = [ast.literal_eval(j) for j in test_x]
text_x = np.array(test_x)
test_x = text_x.reshape(418,39,1)

learning_rate = 0.001
total_epoch = 500
batch_size = 128

n_input = 39
n_step = 1
n_hidden = 128
n_class = 2

X = tf.placeholder(tf.float32, [None, n_input, 1])
Y = tf.placeholder(tf.float32, [None, n_class])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))
batch_prob = tf.placeholder(tf.bool)

cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype= tf.float32)
outputs = tf.transpose(outputs, [1,0,2])
outputs = outputs[-1]
model = tf.matmul(outputs, W) + b


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = model, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(len(train_x)/batch_size)

for epoch in range(total_epoch):
    _, cost_val = sess.run([optimizer, cost], feed_dict = {X: train_x, Y: train_y})
    print('Epoch: ', '%04d' % (epoch + 1), 'Avg.cost = ' '{:.3f}'.format(cost_val))

print('Training complete')

prediction = tf.argmax(model, 1)
target = tf.argmax(Y,1)

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("Accuracy", sess.run(accuracy, feed_dict = {X:val_x, Y:val_y}))
test = list(sess.run(prediction, feed_dict = {X: test_x}))

import csv
list1 = [892+i for i in range(len(test))]
list2 = test

d = zip(list1, list2)
with open('output_rnn.csv', 'w',encoding = 'utf8') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(("PassengerId", "Survived"))
    wr.writerows(d)
myfile.close()
