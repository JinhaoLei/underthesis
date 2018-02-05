import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

path = './parameters/fourA/'
predictions = np.load(path + 'predictions.npy')
fig = plt.figure() 
 
hidden = np.load(path + 'hidden.npy')
wanted = hidden[:, -1, :]
print np.max(wanted), np.min(wanted)
classes = [[] for i in range(11)]
for i in range(len(wanted)):
	classes[predictions[i]].append(wanted[i])
#print np.shape(classes[0]), np.shape(classes[1])
#print classes[0]
for i in range(11):
	ax = fig.add_subplot(1,11,i+1)
	ax.set_axis_off()
	im = ax.imshow(np.array(classes[i]).transpose(), cmap=plt.get_cmap('gray'), 
	interpolation='nearest', vmin=-1, vmax=1, aspect='auto') 
	#chartBox = ax.get_position()
	#ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.8, chartBox.height])
#plt.xticks([]), plt.yticks([]) 

#cbaxes = fig.add_axes([0.9, 0.1, 0.03, 0.8]) 
#cb = plt.colorbar(ax1, cax = cbaxes)  
#divider = make_axes_locatable(ax)
#cax = divider.new_vertical(size="5%", pad=0.7, pack_start=True)
#fig.add_axes(cax)
#fig.colorbar(im, cax=cax, orientation="vertical")
fig.colorbar(im, fraction=1.0) 
  
plt.savefig('./plot/hot1.png', format='png')
plt.savefig('./plot/hot1.pdf', format='pdf')