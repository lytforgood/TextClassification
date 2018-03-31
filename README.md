## keras实现深度学习模型 进行文本分类

> 实验数据采用真实邮件数据，涉及个人隐私，无法公开，可自行寻找数据测试--格式为：文本内容,标签
> 模型参数未经过合适调整，目前正在实验修改验证模型当中，修改完成会更新项目

01mail.py 文本数据生成-输出文本 词典 非一次执行

02mail.py 文本词袋向量化/TF-IDF标准化/文本Hash+朴素贝叶斯

03fastText.py fastText库训练

03fastText_keras.py fastText keras实现

04textCNN.py word2vecter做词向量的CNN两种模型 

05textRNN.py 双向lstm随机初始词向量

06textRCNN.py Recurrent Convolutional Neural Networks for Text Classification

07Attention.py 双向LSTM+Attention分层注意网络 -HAN模型 

