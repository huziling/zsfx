import os
import sys
import numpy as np
import pandas as pd
import  random
import matplotlib.pyplot as plt
#线性回归的代码 
#采用随机批量梯度下降




def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def dJ(w, x, y):
    #print(len(y))
    return  x.T.dot(sigmoid(x.dot(w)) - y)/ len(y)
def rmse(w, x, y):
    try:
        return np.sqrt(np.sum((y - sigmoid(x.dot(w))) ** 2)/len(y))
    except:
        return float('inf')
def plot(yp,yr):
    x = [i for i in range(len(yp))]
    plt.scatter(x, yp.T,c = 'r')
    plt.scatter(x, yr.T)
    plt.show()
#迭代100次
#每次次批量为20
#共500批次
def sgd(Y,X,w,iteration = 2000,alpha = 0.1,momentum = 0.9): 
    m = X.shape[0]
    s = [i for i in range(m)]
    for k in range(iteration):
        se = random.sample(s,m)
        gradient = 0
        for i in range(20):
            st = se[i*20:(i+1)*20]
            x = X[st]
            y = Y[st] 
            #print(X.dot(w).T)
            #r = rmse(w,X,Y)
            # 
            gradient = dJ(w,x,y) + momentum * gradient
            w = w - alpha/(10 + k*0.0001) * gradient
    #print(J(w,X,Y))
    #print(X.dot(w).T)]
    return w
        
        


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
    var = x.var(axis = 0)
    mean = x.mean(axis = 0)
    x = (x-mean)/var
    t = np.zeros((l,1))
    k = x.shape[1]
    x = np.append(x,t,axis=1)
    #print(t[0:100,-1])
    for i in range(k):
        x[:,k - i] = x[:,k-i-1]
    x[:,0] = np.float(1)    
    xpos = x[100:300]
    xnga = x[-301:-101]
    xtest = x[:100]
    xtest = np.append(xtest,x[-101:-1],axis = 0)
    ytrain = np.ones((200,1))
    ytrain = np.append(ytrain,np.zeros((200,1)),axis = 0)
    ytest = np.ones((100,1))
    ytest = np.append(ytest,np.zeros((100,1)),axis = 0)
    xtrain = np.append(xpos,xnga,axis = 0)
    w = np.ones((101,1))
    w = sgd(ytrain,xtrain,w)
    ypre = xtest.dot(w)
    #r = rmse(w,xtest,ytest) 
    for i in range(ypre.shape[0]):
        if ypre[i] > 0.5:
            ypre[i] = 1
        else :
            ypre[i] = 0
    t = np.abs(ypre - ytest)
    print(1 - np.sum(t)/200)
    print(ypre.shape,ytest.shape)
    #plot(ypre,ytest)   
    return ypre,1 - np.sum(t)/200


if __name__ == '__main__':
    tag = 200
    r = []
    for i in range(10):
        ntag = tag + i + 1
        mypath = str(ntag)
        #model = load(mypath)y
        path = os.getcwd()
        xpath = os.path.join(path,'document')
        xpath = os.path.join(xpath,mypath+'f.txt')
        print(xpath)
        vec = np.loadtxt(xpath,delimiter=',')
        ypre,t= init(vec)
        r.append([ypre,t])
    path = os.getcwd()
    ypath = os.path.join(path,'score')
    ypath = os.path.join(ypath,'run.txt')
    with open(ypath,'w') as f:
        for line in r:
            t = ''
            for l in line[0]:
                t += str(l[0])+','
            f.write(t+str(line[1])+'\n')
