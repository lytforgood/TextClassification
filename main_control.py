# -*- coding: utf-8 -*-
"""
使用Python脚本控制多个Python脚本运行
方法一：
直接Python顺序运行
import os
os.system("D:\ProgramData\Anaconda3\python D:\mysoft\dlspace\FastText3.py")  #因为没有环境变量需要制定python路径  mac/linux os.system("python /xx/a.py")
os.system("D:\ProgramData\Anaconda3\python D:\mysoft\dlspace\main_control.py")
方法二：
写成函数形式，调用函数 如下
"""
import dataPreprocess
import FastText
import TextCNNmodel
import TextRNNmodel
import TextRCNNmodel
import TextAttention
import MyModel
print("设置参数")
#获取数据参数
# path = './data/nlpmail_re3.txt'
path="./data/nlpmaildatasample2.csv" #数据输入
w2vpath = "./data/w2c_model"  #w2v模型地址
embedding_dims = 128  # 词向量长度
logpath='./model/mylog.txt' #日志记录地址
modelpath='./model/' #模型保存目录
#模型训练参数
batch_size = 32
epochs = 100
#fastText参数
ngram_range=2
#TextRCNNmodel参数
hidden_dim_1 = 200
hidden_dim_2 = 100


print("获取数据")
x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix=dataPreprocess.getdata(path,embedding_dims,w2vpath)

print("调用模型")
FastText.getdata_train(path,ngram_range,maxlen+10,max_token,embedding_dims,batch_size,epochs,logpath,modelpath,"FastText")
TextCNNmodel.train(x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix,embedding_dims,batch_size,epochs,logpath,modelpath,"TextCNN")
TextRNNmodel.train(x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix,embedding_dims,batch_size,epochs,logpath,modelpath,"TextSimpleRNN")
TextRNNmodel.train2(x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix,embedding_dims,batch_size,epochs,logpath,modelpath,"TextBiLSTM")
TextRNNmodel.train3(x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix,embedding_dims,batch_size,epochs,logpath,modelpath,"TextBiGRU")
TextRCNNmodel.train(x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix,embedding_dims,batch_size,epochs,logpath,modelpath,"TextRCNN",hidden_dim_1,hidden_dim_2)
TextAttention.train(x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix,embedding_dims,batch_size,epochs,logpath,modelpath,"TextAttention")
MyModel.train(x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix,embedding_dims,batch_size,epochs,logpath,modelpath,"MyConBiGRU")
