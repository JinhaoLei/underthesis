import matplotlib.pyplot as plt
import numpy as np

x = [1,2,3,4,5]
xlabels = ['100%', '33%', '16.5%', '5.0%', '1.6%']
s0= {'MLP':[53.67,41.48,32.48,27.55,21.94], 'RNN':[68.48,66.7,63.03,49.91,30.27],
      'WDCNN':[67.67, 58.48, 41.48, 31.7, 22.76], 'LSTM':[86.12,70.3,64.79,54.36,36.27]}
s1 = {'MLP':[51.48,49.52,44.61,37.21,27.58], 'RNN':[59.12,46.76,39.97,25.03,24.3],
      'WDCNN':[59.7, 52.03,41.24,27.55,23.91], 'LSTM':[66.76,59, 57.3, 47.64,38.48]}
s2 = {'MLP':[59.48,35.76,19.61,15.3, 12.79], 'RNN':[73.39,55.94,48.73,39.73,28.36],
      'WDCNN':[84.67,72.52,61.55,42, 36.73], 'LSTM':[83.06,81.73,72.09,50.55,37.91]}
s3 = {'MLP':[53.48,28.61,12.85,11.61,12.03], 'RNN':[82.03,53.39,24.52,25.24,15.09],
      'WDCNN':[87.79,78.79,52.18,20.24,21.33], 'LSTM':[79.94,74.88,61.79,50.21,16.52]}

def draw(pos, s0):
	fig = plt.figure()
	ax = fig.add_subplot(111)

	mlp = ax.plot(x[:], s0['MLP'][::-1], 'b-o', label='MLP')
	rnn = ax.plot(x, s0['RNN'][::-1], 'm-v', label='RNN')
	wdcnn = ax.plot(x, s0['WDCNN'][::-1], 'g-s', label='WDCNN')
	lstm = ax.plot(x, s0['LSTM'][::-1], 'r-o', label='LSTM')

	chartBox = ax.get_position()
	ax.set_xticks(x)
	ax.set_xticklabels(xlabels[::-1])
	ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.9, chartBox.height])
	ax.legend(('MLP', 'RNN', 'WDCNN', 'LSTM'), 
	bbox_to_anchor=(1,1), loc=2, borderaxespad=0., prop={'size': 10})
	ax.set_yticks([0, 20, 40, 60, 80, 100])
	ax.set_yticklabels([0, 20, 40, 60, 80, 100], fontsize=10)
	ax.set_ylabel('Accuracy(%)', fontsize=10)
	ax.set_xlabel('Training Data', fontsize=10)
	return ax
a = draw(221, s0)
plt.savefig('./plot/weak-s0.png', format='png')
plt.savefig('./plot/weak-s0.pdf', format='pdf')
b = draw(222, s1)

b.set_ylim(20, 80)
plt.savefig('./plot/weak-s1.png', format='png')
plt.savefig('./plot/weak-s1.pdf', format='pdf')
c = draw(223, s2)
plt.savefig('./plot/weak-s2.png', format='png')
plt.savefig('./plot/weak-s2.pdf', format='pdf')
d = draw(224, s3)
plt.savefig('./plot/weak-s3.png', format='png')
plt.savefig('./plot/weak-s3.pdf', format='pdf')



#plt.savefig('./plot/multi-weak-bar.pdf', format='pdf')