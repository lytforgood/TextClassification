#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
gensim的Word2vector使用
pip install gensim
输入数据要求是：分词后数据，以空格为单词的分隔符
原理讲解
https://www.cnblogs.com/f-young/p/7906451.html
"""
from gensim.models import Word2Vec
import pandas as pd
# import cPickle as pickle
# path='./data/nlpmaildata2.pkl'
# f2 = file(path, 'rb')
# d = pickle.load(f2)
# f2.close()

path='./data/nlpmail_re3.txt'
d = pd.read_csv(path,header=None)
d.columns=['title','lable']
# sentences= [str(s).split() for s in sentences]


modelpath="./data/w2c_model"
sentences=list(d["title"])
sentences= [str(s).split() for s in sentences]

model = Word2Vec(sentences, sg=1, size=128,  window=5,  min_count=1,  negative=3, sample=0.001, hs=1, workers=4)
# 1.sg=1是skip-gram算法，对低频词敏感；默认sg=0为CBOW算法。
# 2.size是输出词向量的维数，值太小会导致词映射因为冲突而影响结果，值太大则会耗内存并使算法计算变慢，一般值取为100到200之间。
# 3.window是句子中当前词与目标词之间的最大距离，3表示在目标词前看3-b个词，后面看b个词（b在0-3之间随机）。
# 4.min_count是对词进行过滤，频率小于min-count的单词则会被忽视，默认值为5。
# 5.negative和sample可根据训练结果进行微调，sample表示更高频率的词被随机下采样到所设置的阈值，默认值为1e-3。
#作者在论文中说到，当样本量比较小的时候，选择5-20个negative words效果会比较好，当样本量比较大的时候，2-5个negative words就能得到很好的效果
# 6.hs=1表示层级softmax将会被使用，默认hs=0且negative不为0，则负采样将会被选择使用。
# 7.workers控制训练的并行，此参数只有在安装了Cpython后才有效，否则只能使用单核。
# model["英文"]
model.save(modelpath)
# model = Word2Vec.load(fname)

#模型使用（词语相似度计算等）
# model.most_similar(positive=['woman', 'king'], negative=['man'])
# #输出[('queen', 0.50882536), ...]

# model.doesnt_match("breakfast cereal dinner lunch".split())
# #输出'cereal'

# model.similarity('woman', 'man')
# #输出0.73723527

# model['computer']  # raw numpy vector of a word
#输出array([-0.00449447, -0.00310097,  0.02421786, ...], dtype=float32)
