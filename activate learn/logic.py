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
def sgd(Y,X,w,iteration = 5000,alpha = 0.100,momentum = 0.90): 
    m = X.shape[0]
    print(m)
    s = [i for i in range(m)]
    for k in range(iteration):
        se = random.sample(s,m)
        gradient = 0
        l = int(m / 20)
        for i in range(l):
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
    # var = x.var(axis = 0)
    # mean = x.mean(axis = 0)
    # x = (x-mean)/var
    t = np.zeros((l,1))
    k = x.shape[1]
    x = np.append(x,t,axis=1)
    #print(t[0:100,-1])
    for i in range(40):
        x[:,k - i] = x[:,k-i-1]
    x[:,0] = np.float(1)   
    train = [290+i for i in range(20)]
    train =  set(train)
    xtest = x[:300]
    xtest = np.append(xtest,x[-301:-1],axis = 0)    
    ytest = np.ones((300,1))
    ytest = np.append(ytest,np.zeros((300,1)),axis = 0)
    allx = [i for i in range(600)]
    allx = set(allx) - train
    print(len(allx))
    ll = 0
    beasta = 0
    beastp = None
    beasti = 0
    for ii in range(116):
        print(ii)
        l1 = 0
        l2 = 0
        for t in train:
            if t >= 300:
                l2+=1
            else:
                l1 += 1
        ytrain = np.ones((l1,1))
        ytrain = np.append(ytrain,np.zeros((l2,1)),axis = 0)
        xtrain = x[sorted(list(train))]
        w = np.ones((101,1))
        w = sgd(ytrain,xtrain,w)
        ypre = sigmoid(xtest.dot(w))
        #r = rmse(w,xtest,ytest)
        temp = ypre.copy()
        for i in range(ypre.shape[0]):
            if ypre[i] > 0.5:
                ypre[i] = 1
            else :
                ypre[i] = 0
        t = np.abs(ypre - ytest)
        print(1 - np.sum(t)/600)
        print(ypre.shape,ytest.shape)
        if 1 - np.sum(t)/600 > 0.8:
            return ypre,1 - np.sum(t)/600,ii
        if 1 - np.sum(t)/600 > beasta:
            beasta = 1 - np.sum(t)/600
            beastp = ypre
            beasti = ii
        lastset = allx - train
        temp = temp.T[0]
        # print(temp)
        # print(temp[list(lastset)].shape)
        
        er = np.abs(temp-0.5)
        #print(er)
        nt = np.argsort(er)
        pin = 0
        for t in nt:
            if t not in train:
                train.add(t)
                pin += 1
            if pin == 5:
                break
        #print(nt)
        #print(nt)
        #print(nt.shape)
        print(len(train),len(allx))
        #input()
    #plot(ypre,ytest)   
    return beastp,beasta,beasti


if __name__ == '__main__':
    tag = 200
    r = []
    for i in range(10):
        r = []
        ntag = tag + i + 1
        mypath = str(ntag)
        #model = load(mypath)y
        path = os.getcwd()
        xpath = os.path.join(path,'document')
        xpath = os.path.join(xpath,mypath+'f.txt')
        print(xpath)
        vec = np.loadtxt(xpath,delimiter=',')
        ypre,t,ll = init(vec)
        r.append([ypre,t])
        path = os.getcwd()
        ypath = os.path.join(path,'run'+mypath+'.txt')
        with open(ypath,'w') as f:
            f.write(str(ll*5+20)+'\n')
            for line in r:
                t = ''
                for l in line[0]:
                    t += str(l[0])+','
                f.write(t+'\n')
                f.write(t+str(line[1])+'\n')
