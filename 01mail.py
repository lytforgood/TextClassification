#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
文本数据生成 输出 文本 词典
思考：
去停用词改为使用信息增益、互信息法、L1正则化选择词特征
'''
import pandas as pd
import re
import jieba
import cPickle as pickle
import sys #这里只是一个对sys的引用，只能reload才能进行重新加载


def data2all():
    ##数据生成
    f1 = pd.read_csv("./data/data2016_0730_1028.csv",sep=',',header=None,encoding="utf-8")
    f2 = pd.read_csv("./data/data2016_0730_1028.csv",sep=',',header=None,encoding="utf-8")
    f3 = pd.read_csv("./data/data20161028_20170108.csv",sep=',',header=None,encoding="utf-8")
    f =pd.concat([f1,f2,f3])
    f.columns = ['accept','title','send','accept','time','label','day']

    all_data=f[["title","label"]]

    x=all_data.groupby(["label"])["title"].count()
    # Index([u'个人文件夹(个人过滤器)', u'垃圾箱(系统判断)', u'已退信', u'投递中', u'投递成功', u'收件箱', u'自动转发',u'被拦截(个人过滤器)', u'被拦截(用户黑名单)', u'被拦截(系统拦截)'],
    d1=all_data[(all_data["label"]==x.index[1])].reset_index() #垃圾
    d2=all_data[(all_data["label"]==x.index[4])].reset_index() #投递成功
    d3=all_data[(all_data["label"]==x.index[5])].reset_index() #收件箱

    d=pd.concat([d1,d2,d3])
    d=d[["title","label"]]
    d.to_csv("./data/nlpmail.csv",header=False,index=False,encoding="utf-8")

#合并数据
data2all()

stdi,stdo,stde=sys.stdin,sys.stdout,sys.stderr
reload(sys) #通过import引用进来时,setdefaultencoding函数在被系统调用后被删除了，所以必须reload一次
sys.stdin,sys.stdout,sys.stderr=stdi,stdo,stde
sys.setdefaultencoding('utf-8')

##读取文件
d = pd.read_csv("./data/nlpmail.csv",sep=',',header=None,encoding="utf-8")
d.columns = ['title','label']
##类别编码
def label2num(x):
  l=0
  if(x==u"垃圾箱(系统判断)"):
    l=1
  return l
d["label2"]=[label2num(x) for x in d["label"]]
d["index"]=range(d.shape[0])


##去除标点符号
def remove_punctuation(line):
    #中文标点 ！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.
    #英文标点 !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
    try:
      line = re.sub("[！？。｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+".decode("utf-8"), "",line.decode("utf-8"))
    except Exception as e:
      print "error"
    return line

##结巴分词
def cutline(line):
    line=str(line) #防止只有数字的识别为float  纯数字转换成 数字 一词
    words = jieba.cut(line, cut_all=False)
    re=" ".join(words)
    return re

#创建字典 词级别
def createVocabList(dataSet):
    all_data=[]
    for line in  dataSet:
      for words in line.split(" "):
        all_data.append(words)
    all_data=set(all_data)
    return all_data
#去标点
d["title"]=[remove_punctuation(x) for x in d["title"]]
d=d[["index","title","label2"]]
#分词
d["title"]=[cutline(x) for x in d["title"]]

##保存文本
path='./data/nlpmaildata.pkl'
output = file(path, 'wb')
pickle.dump(d, output, True)
output.close()
# ##保存字典
# vocab_dir=createVocabList(d["title"])
# vocab_dir=list(vocab_dir)
# path='./data/vocab_dir.pkl'
# output = file(path, 'wb')
# pickle.dump(vocab_dir, output, True)
# output.close()

#数据清洗 替换英文和字母 选取文本长度>4的文本
def replayxx(line):
    words=line.split(" ")
    newwords=[]
    for w in words:
      if  w.encode( 'UTF-8' ).isdigit():
        w="数字"
      if  w.encode( 'UTF-8' ).isalpha():
        w="英文"
      if  re.match('^[A-Za-z0-9]+$',w):
        w="数字英文"
      newwords.append(w)
    res=" ".join(newwords)
    return res
d["title"]=[replayxx(x) for x in d["title"]]

d=d[["title","label2"]].reset_index(drop = True)
d.columns=['title','lable']

path='./data/nlpmaildata2.pkl'
output = file(path, 'wb')
pickle.dump(d, output, True)
output.close()
#切分数据集
df1=d[(d["label2"]==1)].sample(frac=0.2)
df2=d[(d["label2"]==0)].sample(frac=0.2)
d=pd.concat([df1,df2])
from sklearn.utils import shuffle
d = shuffle(d)
d =d.sample(frac=0.2)
path='./data/nlpmaildatasample2.pkl'
output = file(path, 'wb')
pickle.dump(d, output, True)
output.close()

d.to_csv("./data/nlpmaildatasample2.csv",header=False,index=False)
