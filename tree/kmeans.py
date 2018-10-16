#import minpy.numpy as np

import numpy as np
#import torch
import gensim
import time
import pandas as pd
# model = gensim.models.Word2Vec.load("w2v_model1")
# data = None
# i  = 0
# with open("shiti.txt",'r',encoding="UTF-8") as f:
#     for line in f:
#         line = line.strip()
#         als = line.split('_')
#         temp = np.ones((1,100))*0.01
#         i += 1
#         if i % 1000 == 0:
#             print(i)
#         for w in als:
#             #tt = np.ndarray(list(model[w]))
#             if w in model:
#                 temp = temp + model[w].reshape((1,100))
#         if np.sum(temp) == 0:
#             print("catch",i,als)
#         if data is None:
#             data = temp
#         else:
#             data = np.append(data,temp,axis = 0)
#             #data = np.vstack((data,temp))
# data = np.mat(data)
# np.savetxt("dataset.txt",data,fmt='%0.8f',delimiter=',')



    




def yuxian(vecA,vecB):
    #st = time.time()
    num = (vecA * vecB.T)[0,0] #若为行向量则 A * B.T  
    denom = np.linalg.norm(vecA) * np.linalg.norm(vecB)
    if denom == 0:
        print(vecA,vecB)
    cos = num / denom #余弦值  
    sim = 1 - num /denom #归一化 
    #print(time.time() - st) 
    return sim
def jyuxian(A,B):
    num = A * B.T
    demo = np.linalg.norm(A,axis=1) * np.linalg.norm(B,axis=1)
    print(num.shape,demo.shape)
    #input() 

    

def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2))) #la.norm(vecA-vecB)

def randCent(dataSet, k):
    n = dataSet.shape[1]
    centroids = np.mat(np.zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = np.min(dataSet[:,j]) 
        rangeJ = np.float((np.max(dataSet[:,j]) - minJ))
        centroids[:,j] = np.mat(minJ + rangeJ * np.random.rand(k,1))
    return centroids
    
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = dataSet.shape[0]
    clusterAssment = np.mat(np.zeros((m,2)))#create mat to assign data points 
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    l = 0
    print(m)
    while clusterChanged:
        clusterChanged = False
        l+=1
        if l == 100:
            print("ERROR")
        if l == m:
            print("dead")
        #st = time.time()
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = np.inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,0] = minIndex
            clusterAssment[i,1] = minDist
        #print centroids
        #print(time.time() - st)
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0]==cent)[0]]#get all the point in this cluster
            centroids[cent,:] = np.mean(ptsInClust, axis=0) #assign centroid to mean 
    print("donw one")
    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeas=distEclud):
    m = dataSet.shape[0]
    clusterAssment = np.mat(np.zeros((m,2)))
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0] #create a list with one centroid
    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(np.mat(centroid0), dataSet[j,:])
    while (len(centList) < k):
        lowestSSE = np.inf
        for i in range(len(centList)):
            st = time.time()
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas = distMeas)
            sseSplit = np.sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0],1])
            print ("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss
                lowestSSE = sseSplit + sseNotSplit
            print("one step",time.time() - st)
        bestClustAss[np.nonzero(bestClustAss[:,0] == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[np.nonzero(bestClustAss[:,0] == 0)[0],0] = bestCentToSplit
        print ('the bestCentToSplit is: ',bestCentToSplit)
        print ('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return np.mat(centList), clusterAssment

def eva(data,centList,clusterAssment):
    y = []
    for i in range(4):
        xd = data[np.nonzero(clusterAssment[:,0].A == i)[0],:]
        ce = centList[i]
        jyuxian(xd,ce)
        d = clusterAssment[np.nonzero(clusterAssment[:,0].A == i)[0],:]
        y.append(d.mean(axis = 0)[0,1])
    print(y)
    return y


# data = np.loadtxt("dataset.txt",delimiter=',')
# #print(type(data))
# #print(data.shape)
# data = np.mat(data)


data = np.loadtxt("dataset.txt",delimiter=',')
var = data.var(axis = 0)
mean = data.mean(axis = 0)
data = (data-mean)/var
print(type(data))
print(data.shape)
data = np.mat(data)
res1 ,res2 = biKmeans(data,4,yuxian)
print(res2)
print(res1.shape)
print(res2.shape)
y = eva(data,res1,res2)
with open('res.txt','w') as f:
    i = 0
    for l in res2:
        f.write(str(i)+','+str(l[0,0])+'\n')
        i += 1
with open('eva.txt','w') as f:
    for i in y:
        print(i)
        f.write(str(i)+'\n')