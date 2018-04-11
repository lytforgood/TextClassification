## keras实现深度学习模型 进行文本分类

> 实验数据采用真实邮件数据，涉及个人隐私，无法公开，可自行寻找数据测试--格式为：文本内容,标签

> 模型参数未经过合适调整，目前正在实验修改验证模型当中，修改完成会更新项目

- py2 详见py2目录下说明
- main_control.py 主程序入口
- dataPreprocess.py 数据处理 数据输入为：句子中的词(空格分开),标签
- word2vec.py 训练word2vec模型
- FastText.py fastText keras实现
- TextCNNmodel.py word2vecter做词向量的CNN模型 
- TextRNNmodel.py SimpleRNN 双向lstm GRU
- TextRCNNmodel.py Recurrent Convolutional Neural Networks for Text Classification
- TextAttention.py 双向LSTM+Attention分层注意网络 -HAN模型 (与论文有区别)
- MyModel.py 并行卷积和双向GRU
