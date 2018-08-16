#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np 
import pickle as pkl 
import csv

seed = 123
tf.set_random_seed(seed)
np.random.seed(seed)

features = np.array([])
y = np.array([])

test_data = np.array([])
test_y = np.array([])
n_test = 0

save_path = "for_scp/"
test_l = [0,4,8]
train_l = [1,2,3,5,6,7,9]
for i in train_l:
	with open(save_path+"features"+str(i),'rb') as f:
		feature = pkl.load(f)
		if i == 0:
			print(feature.shape()[1])
			features = feature
		else:
			print(features.shape()[1],feature.shape()[1])
			features = np.concatenate((features,feature),axis=0)
	with open(save_path+"labels"+str(i),'rb') as f:
		y_r = pkl.load(f)
		if i == 0:
			y = y_r
		else:
			y = np.concatenate((y,y_r),axis=0)
for i in test_l:
	with open(save_path+"features"+str(i),'rb') as f:
		feature = pkl.load(f)
		if i == 0:
			test_data = feature
		else:
			test_data = np.concatenate((test_data,feature),axis=0)
	with open(save_path+"labels"+str(i),'rb') as f:
		y_r = pkl.load(f)
		if i == 0:
			test_y = y_r
		else:
			test_y = np.concatenate((test_y,y_r),axis=0)

x = tf.placeholder(tf.float32)
labels = tf.placeholder(tf.float32)

mini_batch_size = 10000
mini_batches = [features[k:k+mini_batch_size] for k in range(0,len(features))]

learning_rate = 0.01
hidden_num = [features.shape[1],1000,100,10,2]

b_shape = []
w_shape = []
init_range = []

w = [] 
b = []
activates = []
zx = [x]

for i in range(len(hidden_num)-1):
	b_shape.append(hidden_num[i+1])
	w_shape.append(hidden_num[i],hidden_num[i+1])
	init_range.append(np.sqrt(6.0/(w_shape[i][0])+w_shape[i][1]))

	b.append(tf.Variable(tf.zeros(b_shape[i],dtype=tf.float32)))
	w.append(tf.Variable(tf.random_uniform(w_shape[i],minval=-init_range[i],maxval=init_range[i],dtype=float32)))
	zs.append(tf.matmul(zs[i],w[i]))
	if i != len(hidden_num)-2:
		activates.append(tf.nn.relu(zs[-1]+b[i]))
predict = tf.nn.softmax(zs[-1]+b[-1])

loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=predict,labels=labels))/mini_batch_size
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
epochs = 100

for i in range(epochs):
	for mini_batch in mini_batches:
		sess.run(train_step,feed_dict={x:mini_batch,labels:y})
	predict_right_tf = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(predict,1),tf.argmax(labels,1)),"float"))
	predict_right = sess.run(predict_right_tf,feed_dict={x:test_data,labels:test_y})
	print("Epochs {0}:{1} / {2}".format(i, predict_right, n_test))















