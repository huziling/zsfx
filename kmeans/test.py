import pandas as pd 
import numpy as np
import jieba
#用jieba去分词

#载入停用词
stop = []
with open("stopwords.txt",'r',encoding = "UTF-8") as f:
    for line in f:
        stop.append(line.strip())
jieba.enable_parallel(8)
dframe = pd.read_excel("final_train.xlsx")
#读入所有的content，用jieba分词后储存词向量
with open("out.txt",'w',encoding = "UTF-8") as f:
    for line in dframe['content']:
        wordlist = list(jieba.cut(line.strip('\n')))
        ostr = ""
        for word in wordlist:
            if  len(word) > 1 and word not in stop:
                ostr += word
                ostr += " " 
        f.write(ostr.strip() + "\n")
ss = set([])
print("down1")
#去掉重复的实体
df = pd.read_excel("final_train.xlsx")
for row in df.iterrows():
    if row[1]['e1'] is np.nan or row[1]['e2'] is np.nan:
        continue
    allsam = row[1]['e1']+'_'+row[1]['e2'] 
    if allsam not in ss:
        ss.add(allsam)
with open("shiti.txt",'w',encoding = "UTF-8") as f:
    for line in ss:
        f.write(line+'\n')
print("down2")

#用word2vec网络训练网络
import gensim
def getsentence(path):
    sent = []
    with open(path,'r',encoding = "UTF-8") as f:
        for line in f:
            line = line.strip()
            if line != "":
                sent.append(line.split(' '))
    return sent

path = 'out.txt'
sentences = getsentence(path)
model = gensim.models.Word2Vec(sentences,min_count = 5,size = 100,workers =10,window = 5,iter = 100)
model.save("w2v_model1")