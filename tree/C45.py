# coding=utf-8
import math
import operator
import numpy as np
import os
import random
import ast
ctoi = {0:'/location/location/contains',1:"/people/person/nationality",2:'/people/person/place_lived',3:'/business/person/company',4:'NA'}
def loaddata(filename):
	if filename == "":
		return False
	dataset = np.loadtxt(filename,dtype=np.int,delimiter=',')
	dataset = dataset.tolist()
	return  dataset
##计算给定数据集的信息熵
def calcShannonEnt(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key]) / numEntries
		shannonEnt -= prob * math.log(prob, 2)
	return shannonEnt

def splitDataSet(dataSet, axis, value):
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis + 1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet


def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1
	baseEntropy = calcShannonEnt(dataSet)
	bestInfoGainrate = 0.0
	bestFeature = -1
	for i in range(numFeatures):
		featureSet = set([example[i] for example in dataSet])
		newEntropy = 0.0
		splitinfo = 0.0
		for value in featureSet:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet) / float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)
			splitinfo -= prob * math.log(prob, 2)
		if not splitinfo:
			splitinfo = -0.99 * math.log(0.99, 2) - 0.01 * math.log(0.01, 2)
		infoGain = baseEntropy - newEntropy
		infoGainrate = float(infoGain) / float(splitinfo)
		if infoGainrate > bestInfoGainrate:
			bestInfoGainrate = infoGainrate
			bestFeature = i
	return bestFeature


def createTree(dataSet, labels,dp = 0):
	classList = [example[-1] for example in dataSet]
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	bestFeat = chooseBestFeatureToSplit(dataSet)
	if bestFeat == -1:
		return majorityCnt(classList)
	bestFeatLabel = labels[bestFeat]
	myTree = {bestFeatLabel: {}}
	del (labels[bestFeat])
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	#print('down',dp)
	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels,dp+1)
	return myTree



def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]


def classify(inputTree, featLabels, testVec):
	firstStr = list(inputTree.keys())[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	classLabel = None
	for key in secondDict.keys():
		if testVec[featIndex] == key:
			if type(secondDict[key]).__name__ == 'dict':
				classLabel = classify(secondDict[key], featLabels, testVec)
			else:
				classLabel = secondDict[key]
	if classLabel is None:
		classLabel = ctoi[4]
	return classLabel

# -*- coding: cp936 -*-


def getCount(inputTree, dataSet, featLabels, count):
	# global num
	firstStr = inputTree.keys()[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	# count=[]
	for key in secondDict.keys():
		rightcount = 0
		wrongcount = 0
		tempfeatLabels = featLabels[:]
		subDataSet = splitDataSet(dataSet, featIndex, key)
		tempfeatLabels.remove(firstStr)
		if type(secondDict[key]).__name__ == 'dict':
			getCount(secondDict[key], subDataSet, tempfeatLabels, count)
		# 在这里加上剪枝的代码，可以实现自底向上的悲观剪枝
		else:
			for eachdata in subDataSet:
				if str(eachdata[-1]) == str(secondDict[key]):
					rightcount += 1
				else:
					wrongcount += 1
			count.append([rightcount, wrongcount, secondDict[key]])
		# num+=rightcount+wrongcount


def cutBranch_downtoup(inputTree, dataSet, featLabels, count):  # 自底向上剪枝
	# global num
	firstStr = inputTree.keys()[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	for key in secondDict.keys():  # 走到最深的非叶子结点
		if type(secondDict[key]).__name__ == 'dict':
			tempcount = []  # 本将的记录
			rightcount = 0
			wrongcount = 0
			tempfeatLabels = featLabels[:]
			subDataSet = splitDataSet(dataSet, featIndex, key)
			tempfeatLabels.remove(firstStr)
			getCount(secondDict[key], subDataSet, tempfeatLabels, tempcount)
			# 在这里加上剪枝的代码，可以实现自底向上的悲观剪枝
			# 计算，并判断是否可以剪枝
			# 原误差率，显著因子取0.5
			tempnum = 0.0
			wrongnum = 0.0
			old = 0.0
			# 标准误差
			standwrong = 0.0
			for var in tempcount:
				tempnum += var[0] + var[1]
				wrongnum += var[1]
			old = float(wrongnum + 0.5 * len(tempcount)) / float(tempnum)
			standwrong = math.sqrt(tempnum * old * (1 - old))
			# 假如剪枝
			new = float(wrongnum + 0.5) / float(tempnum)
			if new <= old + standwrong and new >= old - standwrong:  # 要确定新叶子结点的类别


				# 误判率最低的叶子节点的类为新叶子结点的类
				# 在count的每一个列表类型的元素里再加一个标记类别的元素。
				wrongtemp = 1.0
				newtype = -1
				for var in tempcount:
					if float(var[1] + 0.5) / float(var[0] + var[1]) < wrongtemp:
						wrongtemp = float(var[1] + 0.5) / float(var[0] + var[1])
						newtype = var[-1]
				secondDict[key] = str(newtype)
				tempcount = []  # 这个相当复杂，因为如果发生剪枝，才会将它置空，如果不发生剪枝，那么应该保持原来的叶子结点的结构
			for var in tempcount:
				count.append(var)
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			continue
		rightcount = 0
		wrongcount = 0
		subDataSet = splitDataSet(dataSet, featIndex, key)
		for eachdata in subDataSet:
			if str(eachdata[-1]) == str(secondDict[key]):
				rightcount += 1
			else:
				wrongcount += 1
		count.append([rightcount, wrongcount, secondDict[key]])  # 最后一个为该叶子结点的类别

def save(Tree, data = [],deep = 0):