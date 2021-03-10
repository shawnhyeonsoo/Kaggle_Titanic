import tensorflow.compat.v1 as tf
import numpy as np
import ast

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.disable_eager_execution()
tf.disable_v2_behavior()

x, y = open('train_boundaries.txt','r',encoding = 'utf8').readlines(), open('train_label.txt','r',encoding ='utf8').readlines()
x, y = [ast.literal_eval(i) for i in x], [ast.literal_eval(j) for j in y]
train_x,train_y = np.array(x), np.array(y)
#train_x,train_y = np.array(x[:802]), np.array(y[:802])
val_x, val_y = np.array(x[802:]), np.array(y[802:])

test_x = open('test_boundaries.txt','r',encoding = 'utf8').readlines()
test_x = [ast.literal_eval(j) for j in test_x]
text_x = np.array(test_x)

X = tf.placeholder(tf.float32, [None, 39])
Y = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)
batch_prob = tf.placeholder(tf.bool)

W1 = tf.Variable(tf.random_uniform([39, 400], -1, 1))
W2 = tf.Variable(tf.random_uniform([400, 800],-1, 1))
W3 = tf.Variable(tf.random_uniform([800,2],-1, 1))

b1 = tf.Variable(tf.zeros([400]))
b2 = tf.Variable(tf.zeros([800]))
b3 = tf.Variable(tf.zeros([2]))

L1 = tf.add(tf.matmul(X,W1), b1)
L1 = tf.layers.batch_normalization(L1, center = True, scale = True, training = batch_prob)
L1 = tf.nn.relu(L1)
L1 = tf.nn.dropout(L1, keep_prob)

L2 = tf.add(tf.matmul(L1, W2), b2)
L2 = tf.layers.batch_normalization(L2, center= True, scale = True, training = batch_prob)
L2 = tf.nn.relu(L2)
L2 = tf.nn.dropout(L2, keep_prob)

L3 = tf.add(tf.matmul(L2, W3), b3)
L3 = tf.layers.batch_normalization(L3, center = True, scale = True, training = batch_prob)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = train_y, logits = L3))
optimizer = tf.train.AdamOptimizer(0.01)
op = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess= tf.Session()
sess.run(init)

for step in range(300):
    sess.run(op, feed_dict= {X: train_x, Y:train_y, keep_prob: 0.9, batch_prob: True})
    if (step + 1)%10 ==0:
        print(step + 1, sess.run(cost, feed_dict = {X:train_x, Y:train_y, keep_prob: 0.9, batch_prob: True}))

prediction = tf.argmax(L3, 1)
target = tf.argmax(Y,1)
#print("Prediction", sess.run(prediction, feed_dict = {X:train_x}))
#print("Target", sess.run(target, feed_dict = {Y: train_y}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("Accuracy", sess.run(accuracy, feed_dict = {X:val_x, Y:val_y, keep_prob: 1, batch_prob: False}))


test = list(sess.run(prediction, feed_dict = {X: test_x, keep_prob: 1, batch_prob: False}))



import csv
list1 = [892+i for i in range(len(test))]
list2 = test

d = zip(list1, list2)
with open('output_hb.csv', 'w',encoding = 'utf8') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(("PassengerId", "Survived"))
    wr.writerows(d)
myfile.close()
