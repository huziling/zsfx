#!usr/bin/python
# -*- coding: utf-8 -*-

import shutil
import gensim  
import os
import sys
import numpy as np

#将glove生成的模型处理成gensim能够使用的模式
#加载模型和分词结果
#每个query对应的词向量数据，保存成文件
def getFileLineNums(filename):  
    f = open(filename,'r',encoding = 'utf-8')  
    count = 0  
    for line in f:  
        count += 1  
    return count

def prepend_line(infile, outfile, line):  
    """ 
    Function use to prepend lines using bash utilities in Linux. 
    (source: http://stackoverflow.com/a/10850588/610569) 
    """  
    with open(infile, 'r',encoding = 'utf-8') as old:  
        with open(outfile, 'w',encoding = 'utf-8') as new:  
            new.write(str(line) + "\n")  
            shutil.copyfileobj(old, new) 
      
def prepend_slow(infile, outfile, line):  
    """ 
    Slower way to prepend the line by re-creating the inputfile. 
    """  
    with open(infile, 'r',encoding = 'utf-8') as fin:  
        with open(outfile, 'w',encoding = 'utf-8') as fout:  
            fout.write(line + "\n")  
            for line in fin:  
                fout.write(line) 

def load(mpath):  
      
    # Input: GloVe Model File  
    # More models can be downloaded from http://nlp.stanford.edu/projects/glove/  
    # glove_file="glove.840B.300d.txt"  
    glove_file = mpath + 'fvectors.txt'  
      
    dimensions = 100  
      
    num_lines = getFileLineNums(glove_file)  
    # num_lines = check_num_lines_in_glove(glove_file)  
    # dims = int(dimensions[:-1])  
    dims = 100  
      
    print(num_lines)  
        #  
        # # Output: Gensim Model text format.  
    gensim_file= mpath + 'fglove_model.txt'  
    gensim_first_line = "{} {}".format(num_lines, dims)  
    print(gensim)
        #  
        # # Prepends the line.  
    #if platform == "linux" or platform == "linux2":  
    prepend_line(glove_file, gensim_file, gensim_first_line)  
    #else:  
    #    prepend_slow(glove_file, gensim_file, gensim_first_line)  
      
        # Demo: Loads the newly created glove_model.txt into gensim API.  
    model=gensim.models.KeyedVectors.load_word2vec_format(gensim_file,binary=False) #GloVe Model  
      
    model_name = gensim_file[6:-4]  
          
    #model.save('/home/qf/GloVe-master/' + model_name)  
      
    return model  




if __name__ == '__main__':  
    tag = 200
    for i in range(10):
        ntag = tag + i + 1
        mypath = str(ntag)
        model = load(mypath)
        path = os.getcwd()
        path = os.path.join(path,'document')
        path = os.path.join(path,mypath+'f.txt')
        vec = None
        k = 0
        with open(mypath+'f.txt','r',encoding = 'utf-8') as f1:
            for line in f1:
                temp = np.zeros((1,100))
                lines = line.strip().split(' ')
                for w in lines:
                    if w in model:
                        temp = temp + model[w].reshape((1,100))
                temp = temp/(np.sqrt(len(lines)) + 0.1)
                if vec is None:
                    vec = temp
                else:
                    vec = np.append(vec,temp,axis = 0)
                k += 1
                if k % 100 == 0:
                    print(k)
        vec = np.mat(vec)
        np.savetxt(path,vec,fmt='%0.8f',delimiter=',')
        # vec = np.loadtxt(path,delimiter=',')
        # print(vec.shape)
        # l = vec.shape[0]
        # k = vec.shape[1]
        # t = np.zeros((l,1))
        # t = np.append(vec,t,axis=1)
        # print(t[0:100])
        # #print(t[0:100,-1])
        # for i in range(k):
        #     t[:,k - i] = t[:,k-i-1]
        # t[:,0] = np.float(1)
        # np.savetxt(path,t,fmt='%0.8f',delimiter=',')


    # myfile='data0/vectors.txt'
    # model = load(myfile)
    
    # ####################################
    # #model_name='model'
    # #model = gensim.models.KeyedVectors.load('data0/'+model_name)  

    # print(len(model.vocab))

    # word_list = [u'celebrating',u'raspberry']  
   
    # for word in word_list:  
    #     print(word,'--' ) 
    #     for i in model.most_similar(word, topn=10):  
    #         print (i[0],i[1])  
    #     print ('')

