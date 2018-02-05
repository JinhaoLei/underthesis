# coding: utf-8
import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt



x=np.linspace(0.5,1.5,1400)

#设置需要采样的信号，频率分量有180，390和600
y=7*np.sin(2*np.pi*180*x) + 2.8*np.sin(2*np.pi*390*x)+5.1*np.sin(2*np.pi*600*x)

yy=np.fft.fft(y)                     #快速傅里叶变换


yf=abs(yy)                # 取绝对值
yf1=abs(yy)/len(x)           #归一化处理


xf = np.arange(len(y))        # 频率
xf1 = xf

f = open('result.csv', 'w')
for i in range(len(xf)):
	f.write(str(xf[i]) + '\t' + str(yf1[i])+'\n')
f.close()