import os 
import numpy as np
import argparse


root = '/home/jdx/dataset/lsh-wt'
#root = '/home/ljh/data/test'
def normal(s):
    #print s
    a = max(s)
    b = min(s)
    #print a, b
    for i in range(len(s)):
        s[i] = (s[i] - b) / float(a - b) 
        #print s[i]

def step(dataset, args):
    slices = []
    stride = int(args.stride)
    #print np.shape(dataset)
    left = 0 
    right = int(args.length)
    while right<= len(dataset):
        slices.append(dataset[left : right])
        left += int(args.stride)
        right = int(args.length) + left
    return slices

def no_step(dataset, args):
    slices = []
    length = int(args.length)
    for i in range(len(dataset)/length):
        slices.append(dataset[i*length: (i+1) * length])
    return slices
def walk(args):
    nodes = os.listdir(root)
    nodes.remove('normal2')
    #leng = []

    channels = args.channel.split()
    num = 0 

    for node in nodes:
        os.system('cd ' + root + '/' + node)
    
        datas = os.listdir(root + '/' + node)
        datas.remove(node + '.tdms.npy')
        for dataname in datas:
            c_data = []
            f = np.load(root + '/' + node + '/' + dataname)
            #print f
            for ch in channels:
                data = f[:, int(ch)]
                #print data
                if args.normalize:
                    normal(data)
                #print data
                c_data.append(data)
            num += 1
            #if num==3:
            #    return
            stack_data = np.stack(c_data, axis = 1)
            total_length = int(args.total_length)
            stack_data = stack_data[:total_length]
            trainset = stack_data[:int(total_length * 0.8)]
            #print trainset
            devset = stack_data[int(total_length * 0.8) : int(total_length * 0.9)]
            testset = stack_data[int(total_length * 0.9):]
            trainset = step(trainset, args)
            devset = no_step(devset, args)
            #print devset
            testset = no_step(testset, args)
            print root + '/' + node + '/' + dataname
            print 'train:'
            print np.shape(trainset)
            print '--------------'
            print np.shape(devset)
            print '--------------'
            print np.shape(testset)
            #f = open('./preprocess_data/'+dataname+'.csv', 'w')
            write(dataname, 'train', trainset)
            write(dataname, 'dev', devset)
            write(dataname, 'test', testset)
            
def write(dataname, type, dataset):
    f = open('./preprocess_data_no_normalize/'+dataname+'_'+type+'.csv', 'w')
    f.write(str(np.shape(dataset)))
    f.write('\n')
    for i in range(np.shape(dataset)[0]):
        for j in range(np.shape(dataset)[1]):
            to_write = [str(k) for k in dataset[i][j]]
            f.write(' '.join(to_write)+'\n')
    f.close()

        #leng.append([root + '/' + node + '/' + data, str(np.shape(f)[0])])
        #os.system('cd ..')
    #break

#f = open('sta_length.csv', 'w')
#for i in leng:
#    f.write('\t'.join(i) + '\n')

#.close()

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--stride', dest='stride', default=5000)
    parser.add_argument('--length', dest='length', default=5000)
    parser.add_argument('--channel', dest='channel', default='0 1 2 3')
    parser.add_argument('--normalize', dest='normalize', default=False)
    parser.add_argument('--total_length', dest='total_length', default=5040000)
    args = parser.parse_args()
    walk(args)

if __name__ == '__main__':
    main()
