import jieba
import numpy as np
import os
import sys
jieba.enable_parallel(6)
stopwords = [line.strip() for line in open('stopwords.txt', 'r', encoding='utf-8').readlines()] 
def fen(arttxt,stopwords):
    txt  = jieba.cut(arttxt.strip())
    #print('this 4',artti,ti)
    #print('this 5',arttxt,txt)
    otxt = ''
    for t in txt:
        if len(t) > 1 and t not in stopwords:
            otxt += t + ' '
    return otxt
tag = 200
for i in range(10):
    ntag = tag + i + 1
    mypath = str(ntag)
    path = os.getcwd()
    path = os.path.join(path,'document')
    path = os.path.join(path,mypath+'.txt')
    vec = None
    k = 0
    with open(mypath+'f.txt','w',encoding = 'utf-8') as f1, open(path,'r',encoding = 'utf-8') as f2:
            for line in f2:
                lines = line.strip().split(' ')
                line = fen(line,stopwords)
                f1.write(line+'\n')
    

import os
import sys

mpath = 'data'
workpath = os.getcwd()

# #创建文件夹，提取每个query的res信息
# os.makedirs(path)
# i = 0
# f2 = open(os.path.join(path,'qes.txt'),'w',encoding = 'utf-8')
# with open("Hiemstra_LM0.15_Bo1bfree_d_3_t_10_16.res",'r',encoding = 'utf-8') as f1:
#     for line in f1:
#         lines = line.strip().split(' ')
#         oline = lines[2]+' '+lines[3]+' '+lines[4]+'\n'
#         f2.write(oline)
#         i += 1
#         if i % 10000 == 0 and i != 0:
#             path = os.path.join(workpath,mpath+str(int(i/10000)))
#             os.makedirs(path)
#             f2.close()
#             f2 = open(os.path.join(path,'qes.txt'),'w',encoding = 'utf-8')
#         if i == 100000:
#             break
# f2.close()





##为每个query的每个article读取标题和body
# 并进行分词
#分词后存储，每一个article 按照id存储一个文件
#用两个文件分别存储所有的title 分词 和 body分词
# from bs4 import BeautifulSoup
# import jieba
# jieba.enable_parallel(6)
# stopwords = [line.strip() for line in open('stopwords.txt', 'r', encoding='utf-8').readlines()] 
# def fen(artti,arttxt,stopwords):
#     ti = jieba.cut(artti.strip())
#     txt  = jieba.cut(arttxt.strip())
#     #print('this 4',artti,ti)
#     #print('this 5',arttxt,txt)
#     oti = ''
#     otxt = ''
#     for t in ti:
#         if len(t) > 1 and t not in stopwords:
#             oti += t + ' '
#     for t in txt:
#         if len(t) > 1 and t not in stopwords:
#             otxt += t + ' '
#     return oti,otxt
# for k in range(10):
#     dpath = os.path.join(workpath,mpath+str(int(k)))
#     with open(os.path.join(dpath,'qes.txt'),'r',encoding = 'utf-8') as f1:
#         soup = BeautifulSoup(open(os.path.join(dpath,'ar.txt'),'r',encoding = 'utf-8'),'lxml')
#         aid = f1.readline().split(' ')[0]
#         f2 = open(os.path.join(dpath,'title.txt'),'w',encoding = 'utf-8')
#         f3 = open(os.path.join(dpath,'body.txt'),'w',encoding = 'utf-8')
#         artti = ''
#         arttx = ''
#         i = 0
#         for art in soup.find_all('article'):
#             artid = art.article_id.get_text().strip()
#             #print(art..get_text())
#             #print(artid,aid,artid == aid)
#             #input()
#             artti += art.title.get_text()+' '
#             arttx += art.txt.get_text()+' '
#             i+=1
#             #print('this 3',artti, arttx)
#             ti,body = fen(artti,arttx,stopwords)
#             #print(ti,body)
#             #input()
#             tpath = os.path.join(dpath,artid)
#             if os.path.exists(tpath) == False:
#                 os.makedirs(tpath)
#             f = open(os.path.join(tpath,'ti.txt'),'w',encoding = 'utf-8')
#             f.write(ti+' ')
#             f2.write(ti+'\n')
#             f.close()
#             f = open(os.path.join(tpath,'body.txt'),'w',encoding = 'utf-8')
#             f.write(body+' ')
#             f.close()
#             f3.write(body+'\n')
#             if i % 100 == 0:
#                 print('down',i)
#         print('down',i)
#         f2.close()
#         f3.close()
                