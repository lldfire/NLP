#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-23 15:23:08
# @Author  : lldpy (1723031704@qq.com)
# @Link    : https://github.com/lldfire/
# @Version : $Id$

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 实现步骤
# 1、语料处理：预处理、处理合适的语料；
#     1、需要分词处理
#     2、不需要去除停用词（word2vec算法依赖上下文）
# 2、训练模型
# 3、
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# from gensim.models import Word2Vec
# from gensim.models import word2vec


# 读取已经分好词的语料
# sentences = word2vec.Text8Corpus('../datas/sentences.txt')

# model = Word2Vec(
#     sentences,    # 训练语料库，可迭代列表
#     sg=1,    # =1是skip-gram算法，默认=0是cbow算法
#     size=100,    # 输出词向量维数，默认100，取值一般在100-200之间
#     window=5,    # 一个句子中当前词语和预测词语之间的最大距离，默认5
#     min_count=5,    # 忽略低于此频率的词语，默认5
#     # 以下3个参数为采样和学习率相关的，可不设置
#     # negative=3,    #
#     # sample=0.001,    # 表示更高频率的词被随机下采样到所设置的阈值，默认值为1e-3
#     # hs=1,    # 层级softmax将会被使用，默认hs=0且negative不为0，则负采样将会被选择使用
#     workers=4    # 训练向量时，使用的线程数，默认为计算机核数
#     )
# model.save('./sanguoyanyi.model')

# =============================================================================
# 加载模型


# 初始化模型
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models import KeyedVectors


model = Word2Vec.load('./sanguoyanyi.model')

# print(model.wv['刘备'])    # 查看词向量

# print(model.wv.similarity("刘备", "司马懿"))    # 计算两个词向量的余弦值

# 计算余弦值最近的10个词或topn个词
# print(model.wv.most_similar("孙权"))
print(model.wv.similar_by_word('刘备', topn=100))   # 最接近的100个词

# 将模型的词向量保存至本地
# path = get_tmpfile("wordvectors.kv")
# model.wv.save(path)

# wv = KeyedVectors.load("./sanguoyanyi.model", mmap='r')
# vector = wv['刘备']  # numpy vector of a word