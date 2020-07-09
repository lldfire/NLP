import logging
# import os
import pickle
import random
from collections import defaultdict

import hanlp

from pyhanlp import *
from AveragedPerceptron import AveragedPerceptron


class PerceptronTagger:
    # 每个句子向前和向后两个词，协助统计句子的前两个词和后两个词的词性
    START = ['-START-', '-START2-']
    END = ['-END-', '-END2-']

    def __init__(self):
        self.model = AveragedPerceptron()
        self.tagdict = {}     # 词性标注词典
        self.classes = set()  # 词性类别
        # 分词系统
        self.tokenizer = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')
        # 是否加载模型

    def tag(self, corpus: str) -> list:
        """ 标记所给语料的词性 """
        # 按换行符分割语料为句子，按空格切分句子为词语
        # 这里是中文，不能使用空格分词，需要使用分词模块分词
        def s_split(t): return t.split('\n')   # 以换行符分割句子
        def w_split(s): return self.tokenizer(s)    # 此处需要改

        def split_sents(corpus):
            for s in s_split(corpus):
                yield w_split(s)

        prev, prev2 = self.START
        tokens = []
        for words in split_sents(corpus):
            context = self.START + \
                [self._normlize(w) for w in words] + self.END
            for i, word in enumerate(words):
                tag = self.tagdict.get(word)
                # 词性词典中该无改词的词性时，使用模型预测
                if not tag:
                    features = self._get_features(i, word, context, prev, prev2)
                    tag = self.model.predict(features)
                tokens.append((word, tag))
                # 修改当前词的前两个、一个词性
                prev2 = prev
                prev = tag
        return tokens

    def train(self, sentences, save_loc=None, nr_iter=5):
        """
        :params sentences: 待训练的语料
        :params save_loc: 保存地址
        :params nr_iter: 模型训练的迭代次数
        :return :None
        """
        self._make_tagdict(sentences)
        self.model.classes = self.classes
        for iter_ in range(nr_iter):
            c = 0
            n = 0
            for words, tags in sentences:
                prev, prev2 = self.START
                context = self.START + \
                    [self._normlize(w) for w in words] + self.END
                for i, word in enumerate(words):
                    # 首先从词典中获取词性，否则使用模型预测
                    guess = self.tagdict.get(word)
                    if not guess:
                        feats = self._get_features(
                            i, word, context, prev, prev2)
                        guess = self.model.predict(feats)
                        self.model.update(tags[i], guess, feats)
                    prev2 = prev
                    prev = guess

                    # 预测值与真实值相同的次数
                    c += guess == tags[i]
                    n += 1
            random.shuffle(sentences)
            logging.info(
                "Iter {0}: {1}/{2}={3}".format(iter_, c, n, _pc(c, n)))
        self.model.average_weights()
        if save_loc is not None:
            # 保存模型时参数的顺序，特征权重，词性词典，词性集合
            pickle.dump(
                (self.model.weights, self.tagdict, self.classes),
                open(save_loc, 'wb'), -1
            )
        return None

    def load(self, loc='./data/postag.pic'):
        try:
            w_td_c = pickle.load(open(loc, 'rb'))
        except IOError:
            msg = ("Missing trontagger.pickle file.")
            raise IOError(msg)
        self.model.weights, self.tagdict, self.classes = w_td_c
        self.model.classes = self.classes
        return None

    def _make_tagdict(self, sentences):
        """
        生成词性词典 
        sentences: 整篇语料
        """
        counts = defaultdict(lambda: defaultdict(int))
        for words, tags in sentences:
            for word, tag in zip(words, tags):
                counts[word][tag] += 1
                self.classes.add(tag)    # 向分类的集合中添加词性

        frep_thresh = 20
        ambiguity_thresh = 0.97
        for word, tag_freqs in counts.items():
            # 一个词语的所有词性中频率最高的词性和次数
            tag, mode = max(tag_freqs.items(), key=lambda x: x[1])
            # 所有词性的总数
            n = sum(tag_freqs.values())
            if n >= frep_thresh and (float(mode) / n) >= ambiguity_thresh:
                self.tagdict[word] = tag

    def _normlize(self, word: str):
        """ 修改数字词语的词性 """
        if '-' in word and word[0] != '-':
            return '!HYPHEN'
        elif word.isdigit() and len(word) == 4:
            return '!YEAR'
        elif word.isdigit():
            return '!DIGITS'
        else:
            # 将词语中可能包含的字母全部小写
            return word.lower()

    def _get_features(self, i, word, context, prev, prev2) -> dict:
        """ 构造一个特征 
        :param i: 改词在当前句子中的索引
        :param word: 当前词
        :param context: 当前所在的句子
        :param prev: 当前词的前一个词
        :param prev2: 当前词的前两个词
        """
        def add(name, *args):
            # 构造的特征出现一次，加1
            features[' '.join((name, ) + tuple(args))] += 1
        
        i += len(self.START)    # 抛开前两个占位符
        features = defaultdict(int)
        add('bias')    # 该特征的用处？
        # add('i suffix': word[-3:])  # 英文时单词某位三个字母，不适用于中文
        # add('i pref1': word[0])  # 同样不适用于中文
        add('i-1 tag', prev)    # 前一个的词性
        add('i-2 tag', prev2)    # 前2个词的词性
        add('i tag+i-2 tag', prev, prev2)   # 前1个和前2个词的词性
        add('i word', context[i])    # 当前词语
        add('i-1 tag+i word', prev, context[i])   # 当前词和前一个词的词性
        add('i-1 word', context[i - 1])    # 前一个词
        # add('i-1 suffix', context[i - 1][-3:])   # 不适用中文
        add('i-2 word', context[i - 2])    # 前2个词
        add('i+1 word', word, context[i + 1])   # 后一个词
        # add('i+1 suffix', context[i + 1][-3:])   # 不适用中文
        add('i+2 word', context[i + 2])    # 后两个词
        return features


def _pc(n, d) -> float:
    return (float(n) / d) * 100


if __name__ == "__main__":
    # 语料预处理，将0101中的语料合并为一个txt
    # for i in range(2, 10):
    #     root = f"""/Users/liuliangdong/project/jupyter_project/datasets/nlp_data/2014/010{i}"""
    #     save_path = "/Users/liuliangdong/project/jupyter_project/datasets/nlp_data/rmrb_2014.txt"
    #     txt_list = os.listdir(root)
    #     for name in txt_list:
    #         file_path = os.path.join(root, name)
    #         with open(file_path, 'r', encoding='utf-8') as file:
    #             content = file.read()
    #         with open(save_path, 'a', encoding='utf-8') as sf:
    #             for words in content.split():
    #                 sf.write(words + '\n')

    # 训练模型
    tagger = PerceptronTagger()
    logging.basicConfig(level=logging.INFO)
    # corpus = "/Users/liuliangdong/project/jupyter_project/datasets/nlp_data/rmrb_2014.txt"
    # with open(corpus, 'r', encoding='utf-8') as file:
    #     content = file.readlines()
    # training_data = []
    # sentence = ([], [])
    # for words in content:
    #     # 去掉词语中的方括号
    #     try:
    #         word = words.split('/')[0].strip().replace('[', '').replace(']', '')
    #         tag = words.split('/')[1].strip().replace('[', '').replace(']', '')
    #         sentence[0].append(word)
    #         sentence[1].append(tag)
    #     except IndexError:
    #         sentence = ([], [])
    #         continue
    #     # 一句话的终止
    #     if word == '。':
    #         training_data.append(sentence)
    #         sentence = ([], [])
    #         # break

    # tagger.train(training_data, save_loc='./data/postag.pic')

    # 调用模型
    logging.info('加载模型')
    tagger.load()
    text = """从1月21日美国境内报告首例确诊病例，到现在过去不到半年的时间。十几万鲜活的生命猝然而逝，民众还在疫情漩涡中苦苦挣扎。"""
    
    print('平均感知机', tagger.tag(text))
    
    # hanlp分词模块测试
    # tokenizer = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')
    # text_list = tokenizer(text)
    # tagger_hanlp = hanlp.load(hanlp.pretrained.pos.CTB5_POS_RNN_FASTTEXT_ZH)
    # print('hanlp', tagger_hanlp(text_list))
    
    # pyhanlp
    # Segment = JClas  