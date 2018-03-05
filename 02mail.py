#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
文本向量化 词袋方法 TF-IDF 文本Hash 朴素贝叶斯
'''
import pandas as pd
import re
import jieba
import cPickle as pickle
import numpy as np

##读取文件
# path='./data/nlpmaildata2.pkl'
# path='./data/nlpmaildatasample2.pkl'
# f2 = file(path, 'rb')
# d = pickle.load(f2)
# f2.close()
path='./data/nlpmaildatasample2.csv'
d = pd.read_csv(path,header=None)
d.columns=['title','lable']
#打乱数据
# from sklearn.utils import shuffle
# d = shuffle(d)
#获取停用词表
def get_stopwords(path):
    f= open(path)
    stopwords=[]
    for line in f:
        stopwords.append(line.strip().decode("utf-8"))
    return stopwords
#停用词导入
stopwords=get_stopwords("./data/stopwords.txt")
#获取训练标签
dy=list(d["lable"])
############################################################################################
##方法1.1 自定义词袋方法
##词袋模型
# def bagOfWords2VecMN(vocabList, inputSet):
#     returnVec = [0]*len(vocabList)
#     for word in inputSet:
#         if word in vocabList:
#             returnVec[vocabList.index(word)] += 1
#     return returnVec
# path='./data/vocab_dir.pkl'
# f2 = file(path, 'rb')
# vocab_dir = pickle.load(f2)
# f2.close()
# #转换成list词袋  字典维度太大 会执行失败！！！行数*字典维度/1024/1024/1024=需要多少G内存
# train=[]
# label=list(d["label2"])
# for i in range(len(d["title"])):
#     if(i%10000 ==0):
#         print float(i)/float(len(d["title"]))
#     t=d["title"][i]
#     words=t.split(" ")
#     vec=bagOfWords2VecMN(vocab_dir,words)
#     train.append(vec)
#############################################################################################
##方法1.2 词袋向量化之sklearn
#词袋向量化
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(stop_words=stopwords)
#输入是带空格的分词后list
# d_x=vectorizer.fit_transform(d["title"]).toarray()  #训练并转换
vectorizer.fit(d["title"])
dx=vectorizer.transform(d["title"]).toarray()
#返回满足条件的索引所在位置
# print np.where(d_x[0]>0)
#对应字典获取
vocab_dir=vectorizer.get_feature_names()
#############################################################################################
##方法1.3 词袋向量化之sklearn，TF-IDF和标准化
# from sklearn.feature_extraction.text import TfidfVectorizer
# vector = TfidfVectorizer(stop_words=stopwords)
# vector.fit(d["title"])
# dx=vector.transform(d["title"]).toarray()
# vocab_dir = vector.get_feature_names()#获取词袋模型中的所有词
############################################################################################
##方法2 文本Hash Trick 用哈希技巧矢量化大文本语料库
##原理 hash(文本1)=位置5 hash(文本2)=位置5 位置5的值=1+1or新的哈希函数
# from sklearn.feature_extraction.text import HashingVectorizer
# vectorizer2=HashingVectorizer(n_features = 1000,norm = None,stop_words=stopwords)
# vectorizer2.fit(d["title"])
# dx=vectorizer2.transform(d["title"]).toarray()
#############################################################################################
##朴素贝叶斯按比例验证
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import StratifiedKFold
# clf = MultinomialNB()
# ##修改cv分折方法
# skf = StratifiedKFold(n_splits=5)
# ##修改score
# scores = cross_val_score(clf, dx, dy, cv=skf, scoring='accuracy')
# scores2 = cross_val_score(clf, dx, dy, cv=skf, scoring='f1')
# #评分估计的平均得分和 95% 置信区间由此给出
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# print("f1: %0.2f (+/- %0.2f)" % (scores2.mean(), scores.std() * 2))
#############################################################################################
##按比例切分训练集
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

X_train, X_test, y_train, y_test = train_test_split(dx, dy, test_size=0.2, random_state=0)
# clf = MultinomialNB()
clf = BernoulliNB()
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print("Accuracy: %0.2f" % accuracy_score(y_test, y_pred))
print("F1: %0.2f" % f1_score(y_test, y_pred))

##一致性对比
dtrain=d[0:d.shape[0]/5*3]
dtest=d[d.shape[0]/5*3:d.shape[0]]
X_train, X_test, y_train, y_test=vectorizer.transform(dtrain["title"]).toarray(),vectorizer.transform(dtest["title"]).toarray(),list(dtrain["lable"]),list(dtest["lable"])
clf = BernoulliNB()
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print("Accuracy: %0.2f" % accuracy_score(y_test, y_pred))
print("F1: %0.2f" % f1_score(y_test, y_pred))
# Accuracy: 0.73
# F1: 0.66
# #评价标准
# from sklearn import metrics
# print "Accuracy : %.2f" % metrics.accuracy_score(label, pre_reduce)
# print "recall : %.2f" % metrics.recall_score(label, pre_reduce)
# print "F1 : %.2f" % metrics.f1_score(label, pre_reduce)
