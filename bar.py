import matplotlib.pyplot as plt
import numpy as np

ind = np.linspace(1.0,6.0,5)
width = 0.15

fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(111)


'''svm = [48.12, 45.76, 73.94, 63.06]
mlp = [53.67, 51.48, 59.48, 53.48]
rnn = [68.48, 59.70, 73.39, 82.03]
wdcnn = [67.67, 59.12, 84.67, 87.79]
lstm = [86.12, 66.76, 83.06, 79.94]'''

mlp = [69.18, 60.58, 53.00, 39.67, 23.76]
rnn = [92.30, 82.21, 75.85, 61.21, 46.45]
wdcnn = [83.48, 78.64, 68.73, 44.48, 34.45]
lstm = [94.06, 91.12, 86.42, 69.03, 52.55]


#data = [svm, mlp, rnn, wdcnn, lstm]
data = [mlp, rnn, wdcnn, lstm]
colormap = [plt.cm.tab20(i * 2) for i in range(4)]
b1 = ax.bar(ind + width * 0, data[0][::-1], width, color=colormap[0])
b2 = ax.bar(ind + width * 1, data[1][::-1], width, color=colormap[1])
b3 = ax.bar(ind + width * 2, data[2][::-1], width, color=colormap[2])
b4 = ax.bar(ind + width * 3, data[3][::-1], width, color=colormap[3])
#b5 = ax.bar(ind + width * 4, data[4], width, color=colormap[4])
chartBox = ax.get_position()
ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.9, chartBox.height])
ax.legend((b1, b2, b3, b4), ('MLP', 'RNN', 'WDCNN', 'LSTM'), 
	bbox_to_anchor=(1,1), loc=2, borderaxespad=0., prop={'size': 18})
ax.set_xticks(ind + 2 * width)
ax.set_xticklabels(('1.6%', '5.0%', '16.5%', '33.0%', '100%'), fontsize=18)
ax.set_ylabel('Accuracy(%)', fontsize=18)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_yticklabels([0, 20, 40, 60, 80, 100], fontsize=18)
#ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
plt.savefig('./plot/multi-weak-bar.png', format='png')
plt.savefig('./plot/multi-weak-bar.pdf', format='pdf')