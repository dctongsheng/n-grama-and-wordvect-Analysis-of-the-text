#-*- conding:utf-8 -*-
import jieba
import re
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 数值运算和绘图的程序包
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# 加载机器学习的软件包
from sklearn.decomposition import PCA

#加载Word2Vec的软件包
import gensim as gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import LineSentence

with open("红楼梦.txt",encoding='utf-8') as f:
    text = f.read()
# print(text)
temp = jieba.lcut(text)
words = []
# print(temp)
for i in temp:
    #过滤掉所有的标点符号
    i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：]+", "", i)
    if len(i) > 0:
        words.append(i)
# print(len(words))
# print(words)
trigrams = [([words[i], words[i + 1]], words[i + 2]) for i in range(len(words) - 2)]
# 打印出前三个元素看看
print(trigrams[:3])
# # 得到词汇表
# vocab = set(words)
# print(len(vocab))
# # 两个字典，一个根据单词索引其编号，一个根据编号索引单词
# #word_to_idx中的值包含两部分，一部分为id，另一部分为单词出现的次数
# #word_to_idx中的每一个元素形如：{w:[id, count]}，其中w为一个词，id为该词的编号，count为该单词在words全文中出现的次数
# word_to_idx = {}
# idx_to_word = {}
# ids = 0
#
# #对全文循环，构件这两个字典
# for w in words:
#     cnt = word_to_idx.get(w, [ids, 0])
#     if cnt[1] == 0:
#         ids += 1
#     cnt[1] += 1
#     word_to_idx[w] = cnt
#     idx_to_word[ids] = w
# print(word_to_idx["甄士隐"])


# class NGram(nn.Module):
#
#     def __init__(self, vocab_size, embedding_dim, context_size):
#         super(NGram, self).__init__()
#         self.embeddings = nn.Embedding(vocab_size, embedding_dim)  # 嵌入层
#         self.linear1 = nn.Linear(context_size * embedding_dim, 128)  # 线性层
#         self.linear2 = nn.Linear(128, vocab_size)  # 线性层
#
#     def forward(self, inputs):
#         # 嵌入运算，嵌入运算在内部分为两步：将输入的单词编码映射为one hot向量表示，然后经过一个线性层得到单词的词向量
#         embeds = self.embeddings(inputs).view(1, -1)
#         # 线性层加ReLU
#         out = F.relu(self.linear1(embeds))
#
#         # 线性层加Softmax
#         out = self.linear2(out)
#         log_probs = F.log_softmax(out)
#         return log_probs
#
#     def extract(self, inputs):
#         embeds = self.embeddings(inputs)
#         return embeds
# losses = []  # 纪录每一步的损失函数
# criterion = nn.NLLLoss()  # 运用负对数似然函数作为目标函数（常用于多分类问题的目标函数）
# model = NGram(len(vocab), 10, 2)  # 定义NGram模型，向量嵌入维数为10维，N（窗口大小）为2
# optimizer = optim.SGD(model.parameters(), lr=0.001)  # 使用随机梯度下降算法作为优化器
# # 循环100个周期
# for epoch in range(100):
#     total_loss = torch.Tensor([0])
#     for context, target in trigrams:
#         # 准备好输入模型的数据，将词汇映射为编码
#         context_idxs = [word_to_idx[w][0] for w in context]
#
#         # 包装成PyTorch的Variable
#         context_var = Variable(torch.LongTensor(context_idxs))
#
#         # 清空梯度：注意PyTorch会在调用backward的时候自动积累梯度信息，故而每隔周期要清空梯度信息一次。
#         optimizer.zero_grad()
#
#         # 用神经网络做计算，计算得到输出的每个单词的可能概率对数值
#         log_probs = model(context_var)
#
#         # 计算损失函数，同样需要把目标数据转化为编码，并包装为Variable
#         loss = criterion(log_probs, Variable(torch.LongTensor([word_to_idx[target][0]])))
#
#         # 梯度反传
#         loss.backward()
#
#         # 对网络进行优化
#         optimizer.step()
#
#         # 累加损失函数值
#         total_loss += loss.data
#     losses.append(total_loss)
#     print('第{}轮，损失函数为：{:.2f}'.format(epoch, total_loss.numpy()[0]))
# # 从训练好的模型中提取每个单词的向量
# vec = model.extract(Variable(torch.LongTensor([v[0] for v in word_to_idx.values()])))
# vec = vec.data.numpy()

# 利用PCA算法进行降维
# X_reduced = PCA(n_components=2).fit_transform(vec)
#
#
# # 绘制所有单词向量的二维空间投影
# fig = plt.figure(figsize = (30, 20))
# ax = fig.gca()
# ax.set_facecolor('black')
# ax.plot(X_reduced[:, 0], X_reduced[:, 1], '.', markersize = 1, alpha = 0.4, color = 'white')
#
#
# # 绘制几个特殊单词的向量
# words = ['贾宝玉', '林黛玉', '甄士隐', '贾雨村', '晴雯', '芙蓉花', '牡丹花', '薛宝钗', '王熙凤', '平儿', '贾母', '中秋节']
#
# # 设置中文字体，否则无法在图形上显示中文
# zhfont1 = matplotlib.font_manager.FontProperties(fname='./华文仿宋.ttf', size=16)
# for w in words:
#     if w in word_to_idx:
#         ind = word_to_idx[w][0]
#         xy = X_reduced[ind]
#         plt.plot(xy[0], xy[1], '.', alpha =1, color = 'red')
#         plt.text(xy[0], xy[1], w, fontproperties = zhfont1, alpha = 1, color = 'white')




#
# f = open("红楼梦.txt", encoding='utf-8')
# f = f.read().split("。")
# print(list(f))
# lines = []
# for line in f:
#     temp = jieba.lcut(line)
#     words = []
#     for i in temp:
#         #过滤掉所有的标点符号
#         i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，\- 。？、~@#￥%……&*（）：；‘]+", "", i)
#         if len(i) > 0:
#             words.append(i)
#     if len(words) > 0:
#         lines.append(words)
# print(lines)
# model = Word2Vec(lines, size = 20, window = 2 , min_count = 0)
# renwu = model.wv.most_similar('林黛玉', topn = 20)
# print(renwu)
# rawWordVec = []
# word2ind = {}
# for i, w in enumerate(model.wv.vocab):
#     rawWordVec.append(model[w])
#     word2ind[w] = i
# rawWordVec = np.array(rawWordVec)
# X_reduced = PCA(n_components=2).fit_transform(rawWordVec)
# # 绘制星空图
# # 绘制所有单词向量的二维空间投影
# fig = plt.figure(figsize = (15, 10))
# ax = fig.gca()
# ax.set_facecolor('black')
# ax.plot(X_reduced[:, 0], X_reduced[:, 1], '.', markersize = 1, alpha = 0.3, color = 'white')
#
#
# # 绘制几个特殊单词的向量
# words = ['贾宝玉', '林黛玉', '香菱', '贾政', '晴雯', '妙玉', '袭人', '薛宝钗', '王熙凤', '平儿', '贾母', '探春']
# # 设置中文字体，否则无法在图形上显示中文
# zhfont1 = matplotlib.font_manager.FontProperties(fname='./华文仿宋.ttf', size=16)
# for w in words:
#     if w in word2ind:
#         ind = word2ind[w]
#         xy = X_reduced[ind]
#         plt.plot(xy[0], xy[1], '.', alpha =1, color = 'red')
#         plt.text(xy[0], xy[1], w, fontproperties = zhfont1, alpha = 1, color = 'yellow')
# plt.show()