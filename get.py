from sys import argv
import numpy as np
root = argv[1]
num = int(argv[2])
data = np.load(root)
print np.shape(data)

print np.max(data[:][num])

#f = open('get.csv', 'w')
#for i in a:
#    f.write(str(i)+'\n')

#f.close()



#
