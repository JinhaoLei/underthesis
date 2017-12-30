import os 
import numpy as np
root = '/home/jdx/dataset/lsh-wt'

nodes = os.listdir(root)

leng = []


num = 0 

for node in nodes:
    os.system('cd ' + root + '/' + node)
    #print os.system('pwd')
    datas = os.listdir(root + '/' + node)
    for data in datas:
        f = np.load(root + '/' + node + '/' + data)
        num += 1
        print num
        leng.append([root + '/' + node + '/' + data, str(np.shape(f)[0])])
    os.system('cd ..')
    #break

f = open('sta_length.csv', 'w')
for i in leng:
    f.write('\t'.join(i) + '\n')

f.close()

