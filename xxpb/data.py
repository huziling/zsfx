# -- coding: utf-8 --

#从document 中得到10个query每个文章对应的article的body 
def drop_data():
    filename='data/documents.txt'
    f=open(filename,encoding='UTF-8')
    line=f.readline()
    i=0

    querys=[201,202,203,204,205,206,207,208,209,210]
    pp=0
    f2name='data/document/201.txt'
    f2=open(f2name,'w',encoding='UTF-8')

    import pandas as pd
    df=pd.read_csv('data/score/query201.csv')
    articleid=[]
    for i in df['1']:
        articleid.append(i)

    s=''
    i=0
    templine=''

    tt=0
    while line:
        s=s+' '+line.strip()
        p=s.find(articleid[i])
        #print(p)
        if i+1!=articleid.__len__():
                q = s.find(articleid[i + 1])

        if p>=0:
            m = s.find('<body>')
            n = s.find('</body>')
            if m>=p and n>=m:
                tempb=s[m+6:n]
                print(i,tempb)
                templine+=tempb
                s=s[n+7:]
        elif q>=0:#new article find
            #print(i,templine)
            f2.write(templine+'\n')
            s=s[q:]
            templine=''
            i+=1
            if i== articleid.__len__():
                f2.close()
                pp += 1
                if pp == querys.__len__():
                    print('finish')
                    break
                f2name = 'data/document/' + str(querys[pp]) + '.txt'
                f2 = open(f2name, 'w', encoding='UTF-8')
                import pandas as pd
                df = pd.read_csv('data/score/query' + str(querys[pp]) + '.csv')
                articleid = []
                for k in df['1']:
                    articleid.append(k)
                i = 0
            print('next i',i,s)

        line=f.readline()

        #if i>=1016:
        #    break

        #line = f.readline()


    f.close()
    f2.close()
    print('ok')


#得到每个query 对应的article 和 score
def load_10res():
    qname='data/querys.txt'
    q=open(qname)
    line=q.readline()
    querys=[]
    while line:
        querys.append(line.strip())
        line=q.readline()
    q.close()

    filename='data/Hiemstra_LM0.15_Bo1bfree_d_3_t_10_16.res'
    f=open(filename)
    line=f.readline()

    import pandas as pd
    data=[]
    i=0
    header = ['article_id', 'score']
    while line:
        l=line.strip().split(' ')
        #print(l[0],querys[index])
        if l[0] in querys:
            data.append([l[0],l[2],l[4]])
            #print(l[2],l[4])
            i+=1
        else:
            df = pd.DataFrame(data)
            df=df.drop_duplicates([1],keep='first')
            df.to_csv('data/score/allquery.csv',index=False)
            print(i)
            break

        line=f.readline()

    df=pd.read_csv('data/score/allquery.csv')
    df1 = df.groupby('0')
    for index, data in df1:
        data.to_csv('data/score/query'+str(index)+'.csv', index=False)












