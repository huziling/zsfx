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
    z = sigmoid(x.dot(w))
    y = sigmoid(y)
    return x.T.dot(z - y)/ len(y)
def rmse(w, x, y):
    try:
        return np.sqrt(np.sum((y - x.dot(w)) ** 2)/len(y))
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
def sgd(Y,X,w,iteration = 1000,alpha = 0.05,momentum = 0.9): 
    m = X.shape[0]
    s = [i for i in range(m)]
    for k in range(iteration):
        se = random.sample(s,m)
        gradient = 0
        #print(k)
        for i in range(400):
            st = se[i*20:(i+1)*20]
            x = X[st]
            y = Y[st] 
            #print(X.dot(w).T)
            r = rmse(w,X,Y)
            #print(r)
            # 
            gradient = dJ(w,x,y) + momentum * gradient
            w = w - alpha/(200 + k) * gradient
    #plot(X.dot(w),Y)
    #print(J(w,X,Y))
    #print(X.dot(w).T)]
    return w
        
        


def init(y,x):
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
    w = np.ones((101,1))
    l = int(l*0.8)
    ytrain = y[:l]
    xtrain = x[:l]
    ytest = y[l:]
    xtest = x[l:]
    w = sgd(ytrain,xtrain,w)
    ypre = x.dot(w)
    
    r = rmse(w,xtest,ytest) 
    plot(ypre,y)
    return ypre,r,len(ytest  )


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
        ypath = os.path.join(path,'score')
        ypath = os.path.join(ypath,'query'+mypath+'.csv')
        print(xpath,ypath)
        df = pd.read_csv(ypath)
        vec = np.loadtxt(xpath,delimiter=',')
        y = df['2'].as_matrix()
        y = y.reshape(y.shape[0],1)
        ypre,rmse,l= init(y,vec)
        r.append([rmse,l])
        df['3'] = ypre
        df.to_csv(ypath[:-4] + 'res.txt')
    path = os.getcwd()
    ypath = os.path.join(path,'score')
    ypath = os.path.join(ypath,'run.txt')
    with open(ypath,'w') as f:
        for line in r:
            t = str(line[0]) + ','+str(line[1]) + '\n'
            f.write(t)
