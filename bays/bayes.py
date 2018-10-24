import numpy as np
import re
import random
def createVocabList(dataSet,testlist = None):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)
	if testlist is not None:
		for document in testlist:
			vocabSet = vocabSet | set(document)
	return list(vocabSet)

def bagOfWords2VecMN(vocabList, inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
	return returnVec

def setOfWords2Vec(vocabList,inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else: print("the word: %s is not in my Vocabulary" % (word))
	return returnVec
def trainNB0(trainMatrix,trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory)/float(numTrainDocs)
	p0Num = np.ones(numWords); p1Num = np.ones(numWords)	  #change to ones()
	p0Denom = 2.0; p1Denom = 2.0						#change to 2.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	p1Vect = np.log(p1Num/p1Denom)		  #change to log()
	p0Vect = np.log(p0Num/p0Denom)		  #change to log()
	return p0Vect,p1Vect,pAbusive
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass1)	#element-wise mult
	p0 = np.sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
	if p1 > p0:
		return 1
	else:
		return 0


def textParse(bigString):	#input is big string, #output is word list
	#print(bigString)
	#listOfTokens = re.split(r'\W*', bigString)
	listOfTokens = bigString.split(' ')
	return [tok.lower() for tok in listOfTokens if len(tok) > 2] 

def spamTest():
	docList=[]; classList = []; fullText =[];testlist = []
	# for i in range(1,26):
	# 	wordList = textParse(open('email/spam/%d.txt' %( i)).read())
	# 	docList.append(wordList)
	# 	fullText.extend(wordList)
	# 	classList.append(1)
	# 	wordList = textParse(open('email/ham/%d.txt' % (i)).read())
	# 	#print("%d is ok" %(i))
	# 	docList.append(wordList)
	# 	fullText.extend(wordList)
	# 	classList.append(0)
	classList = np.loadtxt('label.txt',dtype=np.int,delimiter=',')
	classList = classList.tolist()
	with open('data.txt','r',encoding = 'utf-8') as f1:
		for line in f1:
			docList.append(textParse(line.strip()))
	with open('test.txt','r',encoding = 'utf-8') as f1:
		for line in f1:
			testlist.append(textParse(line.strip()))
	print("down read\n")
	#print(docList)
	vocabList = createVocabList(docList,testlist)#create vocabulary
	with open('vocabList.txt','w',encoding = 'utf-8') as f:
		for w in vocabList:
			f.write(w+'\n')
	print("down write v\n")
	trainingSet = list(range(len(docList))); 
	testSet= list(range(len(testlist)))		   #create test set
	# testSet = [] 
	# for i in range(500):
	# 	randIndex = int(random.uniform(0,len(trainingSet)))
	# 	testSet.append(trainingSet[randIndex])
	# 	del(trainingSet[randIndex])
	
	trainMat=[]; trainClasses = []
	for docIndex in trainingSet:#train the classifier (get probs) trainNB0
		trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
		trainClasses.append(classList[docIndex])
	print("down ready\n")
	p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
	errorCount = 0
	res = []
	for docIndex in testSet:		#classify the remaining items
		wordVector = bagOfWords2VecMN(vocabList, testlist[docIndex])
		res.append(classifyNB(np.array(wordVector),p0V,p1V,pSpam))
		# if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
		# 	errorCount += 1
		# 	print ("classification error",docList[docIndex])
	with open('res.txt','w',encoding = 'utf-8') as f:
		for it in res:
			f.write(str(it)+'\n')
	#print( 'the error rate is: ',float(errorCount)/len(testSet))
	#return vocabList,fullText
def calcMostFreq(vocabList,fullText):
	import operator
	freqDict = {}
	for token in vocabList:
		freqDict[token]=fullText.count(token)
	sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
	return sortedFreq[:30]	   

spamTest()