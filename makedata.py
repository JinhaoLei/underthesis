import timefeature as tf
import numpy as np
import os

names = os.listdir('./cnn')
n = 0
for name in names:
	n += 1
	print n 
	data = np.load('./cnn/' + name)
	for i in range(len(data)):
		if i % 10000 == 0:
			print i
		data[i] = tf.timefeature(data[i], 20000)
	np.save('./svm/' + name, data)
