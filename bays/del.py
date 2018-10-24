import jieba
import os
import numpy as np
stop = []
with open("stopwords.txt",'r',encoding = "UTF-8") as f:
    for line in f:
        stop.append(line.strip())
#jieba.enable_parallel(8)

def fenci(line):
    st = ""
    ws = jieba.cut(line.strip())
    for w in ws:
        if len(w) > 1 and w not in stop:
            st += w
            st += ' '
    return st
        
with open("spam_test_no_label.txt",'r',encoding = "UTF-8") as f1,open("test.txt",'w',encoding = 'utf-8') as f2:
    #label = []
    for line in f1:
        #print(line[0])
        #label.append(int(line[0]))s
        #line = line[1:]
        out = fenci(line) 
        f2.write(out + "\n")
    #label = np.array(label)
    #print(label)
    #np.savetxt('label.txt',label,fmt = "%d",delimiter=',')
print('down')
    