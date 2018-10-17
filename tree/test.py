import pandas as pd 
import numpy as np
from math import log
import operator
import random
import gensim
# import jieba

# stop = []
# with open("stopwords.txt",'r',encoding = "UTF-8") as f:
#     for line in f:
#         stop.append(line.strip())
# jieba.enable_parallel(8)
# dframe = pd.read_excel("test.xlsx",header=None)
# with open("out.txt",'w+',encoding = "UTF-8") as f:
#     for line in dframe[3]:
#         print(line)
#         wordlist = list(jieba.cut(line.strip('\n')))
#         ostr = ""
#         for word in wordlist:
#             if  len(word) > 1 and word not in stop:
#                 ostr += word
#                 ostr += " " 
#         f.write(ostr.strip() + "\n")
# ctoi = {'/location/location/contains':0,"/people/person/nationality":1,'/people/person/place_lived':2,'/business/person/company':3}
# df = pd.read_excel("test.xlsx",header=None)
# model = gensim.models.Word2Vec.load("w2v_model1")
# data = None
# for row in df.iterrows():
#     row = row[1]
#     temp1 = np.zeros((1,50))
#     temp2 = np.zeros((1,50))
#     s1 = row[0].split('_')
#     s2 = row[1].split('_')
#     s3 = row[3].split(' ')
#     for w in s1:
#         if w in model:
#             temp1 = temp1 + model[w].reshape((1,50))
#     for w in s2:
#         if w in model:
#             temp2 = temp2 + model[w].reshape((1,50))
#     t = np.zeros((1,1))
#     if s1[0] in s3:
#         t[0,0] = s3.index(s1[0])
#     else:
#         t[0,0] = -1
#     temp2 =  np.append(temp2,t,axis = 1)
#     if s2[0] in s3:
#         t[0,0] = s3.index(s2[0])
#     else:
#         t[0,0] = -1
#     temp2 =  np.append(temp2,t,axis = 1)
  
#     temp1 = np.append(temp1,temp2,axis = 1)
#     if data is None:
#         data = temp1
#     else:
#         data = np.append(data,temp1,axis = 0)
# np.savetxt("testset.txt",data,fmt='%0.8f',delimiter=',')

data = np.loadtxt('testset.txt',delimiter=',')
def lower_bound(nums, target):
    low, high = 0, len(nums)-1
    pos = len(nums) 
    while low<high: 
        mid = int((low+high)/2)
        if nums[mid] < target: 
            low = mid+1 
        else:
            high = mid
            pos = high 
    return pos
l = data.shape[1]
for i in range(l-2):
    col = data[:,i].tolist()
    sc = list(set(col))
    #print(len(col))
    sc.sort()
    lt = int(len(sc)/100)
    sct = []
    for j in range(99):
        sct.append(sc[lt*(1+j)])
    #print(len(sct))
    #input()
    for j in range(len(col)):
        col[j] = lower_bound(sct,col[j])
    col = np.asarray(col)
    data[:,i] = col[:]
np.savetxt('test.txt',data,fmt='%d',delimiter=',')

# data = np.loadtxt('data.txt',delimiter=',')
# ll = [i for i in range(data.shape[0])]
# ll = random.sample(ll,data.shape[0])
# tr = ll[:4000]
# tt = ll[4000:]
# print(len(tr),len(tt))
# train = data[tr]
# test = data[tt]
# np.savetxt('train.txt',train,fmt='%d',delimiter=',')
# np.savetxt('test.txt',test,fmt='%d',delimiter=',')

# dict_name = {1:{2,4},2:{4,5}}
# sortedClassCount = sorted(dict_name.items(), key=operator.itemgetter(1), reverse=True)
# print(sortedClassCount)
# f = open('temp.txt','w')
# f.write(str(dict_name))
# f.close()
 
# #读取
# f = open('temp.txt','r')
# a = f.read()
# dict_name = eval(a)
# f.close()
# print(dict_name)

# import gensim
# def getsentence(path):
#     sent = []
#     with open(path,'r',encoding = "UTF-8") as f:
#         for line in f:
#             line = line.strip()
#             if line != "":
#                 sent.append(line.split(' '))
#     return sent

# path = 'out.txt'
# sentences = getsentence(path)
# model = gensim.models.Word2Vec(sentences,min_count = 5,size = 50,workers =10,window = 5,iter = 100)
# model.save("w2v_model1")