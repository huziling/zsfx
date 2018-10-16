import os
import sys
import numpy as np
import pandas as pd
import  random
import matplotlib.pyplot as plt


def dJ(w, x, y):
    #print(len(y))
    return x.T.dot(x.dot(w) - y)/ len(y)
def J(w, x, y):
    try:
        return np.sqrt(np.sum((y - x.dot(w)) ** 2)/len(y))
    except:
        return float('inf')
def plot(yp,yr):
    x = [i for i in range(len(yp))]
    plt.scatter(x, yp.T,c = 'r')
    plt.scatter(x, yr.T)
    plt.show()

def sgd(Y,X,w,iteration = 1000,alpha = 0.05,momentum = 0.9): 
    m = X.shape[0]
    s = [i for i in range(m)]
    for k in range(iteration):
        se = random.sample(s,m)
        gradient = 0
        for i in range(50):
            st = se[i*20:(i+1)*20]
            x = X[st]
            y = Y[st] 
            #print(X.dot(w).T)
            j = J(w,X,Y)
            #print(j)
            # 
            gradient = dJ(w,x,y) + momentum * gradient
            w = w - alpha/(200 + k) * gradient
    #plot(X.dot(w),Y)
    #print(J(w,X,Y))
    #print(X.dot(w).T)]
    return w
        
        

def LDA(x1,x2,xt):
    mju1 = np.mean(x1, axis=0)#求中心点
    mju2 = np.mean(x2, axis=0)
    cov1 = np.dot((x1 - mju1).T, (x1 - mju1))
    cov2 = np.dot((x2 - mju2).T, (x2 - mju2))
    Sw = cov1 + cov2
    w = np.dot(np.mat(Sw).I,(mju1 - mju2).reshape((len(mju1), 1)))# 计算w
    #print(w.shape)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    y = xt.dot(w)
    y1 = x1.dot(w)
    y2 = x2.dot(w)
    ax.scatter(np.array(range(0, y.shape[0])), y.tolist(), c='r')
    fig.show()
    return y,(mju1.dot(w)+mju2.dot(w))/2
   


def func(x, w):
    return np.dot((x), w)

def init(x):
    i = 0
    l = x.shape[0]
    for line in x:
        if np.sum(line) == 0:
            if i == 0:
                x[i,:] = x[i+1,:] + np.random.randn(1,100)
            else :
                x[i,:] = (x[i-1,:]+x[i+1,:])/2+ np.random.randn(1,100)
        i+=1 
    #var = x.var(axis = 0)
    #mean = x.mean(axis = 0)
    #x = (x-mean)/var
    xpos = x[100:300]
    xnga = x[-301:-101]
    xtest = x[:100]
    xtest = np.append(xtest,x[-101:-1],axis = 0)
    y,u = LDA(xpos,xnga,xtest)
    for i in range(y.shape[0]):
        if y[i] > u:
            y[i] = 1
        else :
            y[i] = 0
    y1 = y[:100]
    y2 = y[100:]
    print(y1.shape,y2.shape)
    t = 0
    for i in y1:
        if i == 1:
            t = t + 1
    for i in y2:
        if i == 0:
            t = t + 1
    t = t / 200
    print(t)
    print(np.sum(y))
    return y,t
    #plot(ypre,y)


if __name__ == '__main__':
    tag = 200
    r = []
    for i in range(10):
        ntag = tag + i + 1
        mypath = str(ntag)
        #model = load(mypath)
        path = os.getcwd()
        xpath = os.path.join(path,'document')
        xpath = os.path.join(xpath,mypath+'f.txt')
        ypath = os.path.join(path,mypath+'res.txt')
        x = np.loadtxt(xpath,delimiter=',')
        y,t= init(x)
        with open(ypath,'w',encoding = 'utf-8') as f:
            for i in y:
                f.write(str(i)+'\n')
            f.write("正确率" + str(t) + '\n') 
    input()
        
    # path = os.getcwd()
    # ypath = os.path.join(path,'score')
    # ypath = os.path.join(ypath,'run.txt')
    # with open(ypath,'w') as f:
    #     for line in r:
    #         t = str(line[0]) + ','+str(line[1]) + '\n'
    #         f.write(t)
