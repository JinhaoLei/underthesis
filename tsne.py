import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_embedding(X, x_min, x_max, num, title=None):
   
   

  

    fig = plt.figure()
    ax = fig.add_subplot(111)
    classeslabel = ['class0', 'class1', 'class2', 'class3', 'class4', 
    'class5', 'class6', 'class7', 'class8', 'class9', 'class10']
    class0=class1=class2=class3=class4=class5=class6=class7=class8=class9=class10=0
    classes = [class0, class1, class2, class3, class4, 
    class5, class6, class7, class8, class9, class10]
    for i in range(11):
        '''plt.text(X[i, 0], X[i, 1], str(predictions[i]),
                 color=plt.cm.Set3(predictions[i] / 11.),
                 fontdict={'weight': 'bold', 'size': 9})'''
        x= np.array(X[i])
        x = (x - x_min) / (x_max - x_min)

        classes[i] = ax.scatter(x[:, 0], x[:, 1], c=plt.cm.hsv((10.0 - i)/ 10.0), label = 'class ' + str(i))
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.9, chartBox.height])
    ax.legend(loc=2,scatterpoints=1, bbox_to_anchor=(1,1), borderaxespad=0., prop={'size': 10})
    #plt.xlabel('After %d Hidden Units'%(20 * (num+1)), fontsize=10)
    plt.xlabel("After Fully-connected Layer", fontsize=10)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

path = './parameters/fourA/'

predictions = np.load(path + 'predictions.npy')

#final = np.load(path + 'final.npy')
#hidden = np.load(path + 'hidden.npy')
#X_embedded = TSNE(init = 'pca').fit_transform(final)
#np.save('fourA.tsne.npy', X_embedded)
to_choose = [(i + 1) * 10 - 1 for i in range(10)]

'''for i in range(10):
    print str(i) + '...'
    maps = [[] for k in range(11)]
    X_embedded = np.load('./plot/fourA-hidden-' + str(i+1) + '.npy')
    x_min, x_max = np.min(X_embedded, 0), np.max(X_embedded, 0)
    for j in range(len(predictions)):
        maps[predictions[j]].append(X_embedded[j])

    plot_embedding(maps, x_min, x_max, i)
    #plt.savefig('./plot/fourA-hidden-' + str(i+1) + '.pdf', format='pdf')
    plt.savefig('./plot/fourA-hidden-' + str(i+1) + '.png', format='png')
    plt.savefig('./plot/fourA-hidden-' + str(i+1) + '.pdf', format='pdf')
    #break 
    #X_embedded = TSNE(init = 'pca').fit_transform(hidden[:,to_choose[i],:])
    #np.save('fourA-hidden-' + str(i + 1) + '.npy', X_embedded)'''
X_embedded = np.load('./plot/fourA.tsne.npy')
x_min, x_max = np.min(X_embedded, 0), np.max(X_embedded, 0)

maps = [[] for k in range(11)]
for j in range(len(predictions)):
    maps[predictions[j]].append(X_embedded[j])
plot_embedding(maps, x_min, x_max, 0)
plt.savefig('./plot/fourA-hidden' + '.png', format='png')
plt.savefig('./plot/fourA-hidden' + '.pdf', format='pdf')

