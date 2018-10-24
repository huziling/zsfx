'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''
from numpy import *
import numpy as np
import operator
from os import listdir
import data

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    # print(sortedDistIndicies)
    # print(labels)
    # input()
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


    
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet#, ranges, minVals

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    # hwLabels = []
    # trainingFileList = listdir('trainingDigits')           #load the training set
    # m = len(trainingFileList)
    # trainingMat = zeros((m,1024))
    # for i in range(m):
    #     fileNameStr = trainingFileList[i]
    #     fileStr = fileNameStr.split('.')[0]     #take off .txt
    #     classNumStr = int(fileStr.split('_')[0])
    #     hwLabels.append(classNumStr)
    #     trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    # testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    # mTest = len(testFileList)
    train_img,train_label,test_img,test_label = data.get_data()
    train_img = train_img.reshape(train_img.shape[0],28*28)
    test_img = test_img.reshape(test_img.shape[0],28*28)
    #train_img[np.where(train_img > 0)] = 1
    #test_img[np.where(test_img > 0)] = 1
    #train_img = autoNorm(train_img)
    #test_img = autoNorm(test_img)
    res = np.zeros((test_img.shape[0] + 1,1))
    mytest = test_img.shape[0]
    for i in range(mytest):
        # fileNameStr = testFileList[i]
        # fileStr = fileNameStr.split('.')[0]     #take off .txt
        # classNumStr = int(fileStr.split('_')[0])
        # vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(test_img[i], train_img, train_label, 4)
        res[i,0] = classifierResult
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, test_label[i]))
        if (classifierResult != test_label[i]): errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mytest)))
    res[-1] = errorCount/float(mytest)
    np.savetxt('res.txt',res,fmt="%.2f",delimiter=',')
    

handwritingClassTest()