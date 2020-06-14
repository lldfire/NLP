import pickle
from collections import defaultdict, Counter
from src.config import get_config


__hmm = None


class Hmm(object):
    """ 根据语料生成hmm模型的三元素,再使用viterbi算法求最优解
        读取语料，处理语料，生成模型，保存模型
    """
    STATE = ['B', 'M', 'E', 'S']
    _words = []
    _states = []
    _vocab = defaultdict()
    _puns = set(u"？?!！·【】、；，。、\s+\t+~@#$%^&*()_+{}|:\"<"
                u"~@#￥%……&*（）——+{}|：“”‘’《》>`\-=\[\]\\\\;',\./■")
    _config = get_config()

    @classmethod
    def read_corpus_from_file(cls, file_path):
        """ 从文件中读取语料 """
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            cls._words.extend([word.strip() for word in line.strip().split() if word.strip() and word.strip() not in cls._puns])

    @classmethod
    def genrate_vocab(cls):
        """ 生成词典，并统计词频 """
        charset = set([w for w in ''.join(cls._words)])
        cls._vocab = {word: 0.0 for word in charset}
        cls._vocab['<UNK>'] = 0.0
        # for word in cls._words:
        #     if word not in cls._vocab:
        #         cls._vocab[word] = 0
        #     cls._vocab[word] += 1

    @classmethod
    def word2state(cls, word):
        """ 将词语转换为隐含状态 """
        word_len = len(word)
        if word_len == 1:
            cls._states.append("S")
        else:
            state = ['M'] * word_len
            state[0] = 'B'
            state[-1] = 'E'
            cls._states.append(''.join(state))

    @classmethod
    def cal_pi_pro(cls):
        """ 计算初始状态分布，将每个单词视为一个状态链 """
        # 统计初始时每个状态的数量，理论上初始状态只有B和S两种，
        pi_state = {k: 0.0 for k in cls.STATE}
        for state in cls._states:
            pi_state[state[0]] += 1
        
        # 统计样本总量
        words_count = len(cls._words)
        # 计算初始状态分布
        pi_pro = {k: v / words_count for k, v in pi_state.items()}
        return pi_pro

    @classmethod
    def cal_A_pro(cls):
        """ 计算状态转移概率 """
        # 统计状态转移的次数
        A_state = {state: {k: 0.0 for k in cls.STATE} for state in cls.STATE}
        states = ''.join(cls._states)
        state_count = Counter(states)    # 统计每个状态的数量
        
        # print(state_count)
        for idx in range(len(states) - 1):
            A_state[states[idx]][states[idx + 1]] += 1
        # print(A_state)
        A_pro = {state: {s: c / state_count[state] for s, c in count.items()}
                 for state, count in A_state.items()}
        return A_pro

    @classmethod
    def cal_B_pro(cls):
        """ 计算观测值发射概率 """
        # 统计每个状态对应的字分布概率
        # 初始化观测值发射矩阵
        B_observe = {k: dict(cls._vocab) for k in cls.STATE}
        
        for index, state in enumerate(cls._states):
            for i, word in enumerate(cls._words[index]):
                # print(i, word, state[i])
                B_observe[state[i]][word] += 1
            # if index == 5:
            #     break
        # 计算观测值转移矩阵
        states = ''.join(cls._states)
        state_count = Counter(states)    # 统计每个状态的数量
        B_pro = {state: {w: c / state_count[state] for w, c in count.items()} 
                 for state, count in B_observe.items()}
        # # 验证结果是否正确：每个状态对应的发射矩阵和为1
        # print(pd.DataFrame(B_pro).sum(axis=0))
        return B_pro

    @classmethod
    def genrate_hmm(cls):
        """ 计算hmm模型参数,并写入本地 """
        for word in cls._words:
            cls.word2state(word)
        pi_pro = cls.cal_pi_pro()
        A_pro = cls.cal_A_pro()
        B_pro = cls.cal_B_pro()
        cls.save_hmm_params(pi_pro, A_pro, B_pro)

    @classmethod
    def save_hmm_params(cls, *args):
        """ 将HMM参数保存至本地 """
        model_path = cls._config.get('segment', 'hmm_model_path')
        with open(model_path, 'wb') as f:
            for params in args:
                pickle.dump(params, f)
        
    @classmethod
    def load_hmm_params(cls):
        """ 加载HMM模型参数 """
        model_path = cls._config.get('segment', 'hmm_model_path')
        hmm_params_key = ['pi_pro', 'A_pro', 'B_pro']
        hmm_params = dict()
        with open(model_path, 'rb') as f:
            for key in hmm_params_key:
                hmm_params[key] = pickle.load(f)
        return hmm_params

    @classmethod
    def train(cls, path=None):
        """ 根据不同语料训练不同的模型 """
        if not path:
            path = cls._config.get('segment', 'train_corpus_path')
        cls.read_corpus_from_file(path)
        cls.genrate_vocab()
        cls.genrate_hmm()
        print('模型训练完成！')

    @classmethod
    def update(cls):
        """ 更新模型参数 """
        pass

    # @classmethod
    # def save_to_file(cls, data, path):
    #     with open(path, 'wb') as f:
    #         pickle.dump(data, f)
    
    # @classmethod
    # def read_to_file(cls, path):
    #     with open(path, 'rb') as f:
    #         data = pickle.load(f)
    #     return data


class Viterbit:

    def __init__(self, hmm_params):
        super().__init__()
        self.pi_pro = hmm_params['pi_pro']
        self.states = self.pi_pro.keys()
        self.A_pro = hmm_params['A_pro']
        self.B_pro = hmm_params['B_pro']

    @staticmethod
    def viterbit_dict(obs, states, pi_pro, A_pro, B_pro):
        path_v = [{}]
        path_max = {}

        # 计算初始时的状态概率
        for state in states:
            path_v[0][state] = pi_pro[state] * B_pro[state].get(obs[0], 0)
            path_max[state] = [state]

        # print(path_max)
        # print(path_v)
        # 计算第一个观测值之后的观测值
        for o in range(1, len(obs)):
            path_v.append({})
            new_path = {}

            # 每个状态到每个观测值的发射概率
            for state in states:
                temp_pro, temp_state = max(((path_v[o - 1][l] * A_pro[l][state] * B_pro[state].get(obs[o], 0), l) for l in states))
                path_v[o][state] = temp_pro

                new_path[state] = path_max[temp_state] + [state]
            path_max = new_path
            # print(path_v)
            # print(path_max)
        best_path_pro, last_state = max((
            path_v[len(obs) - 1][s], s) for s in states)

        return path_max[last_state]

    def cut(self, text):
        best_path = Viterbit.viterbit_dict(
            text, self.states, self.pi_pro, self.A_pro, self.B_pro)
        
        result = ''
        for idx, state in enumerate(best_path):
            if state in ('S', 'E'):
                result += text[idx] + ' '
            else:
                result += text[idx] + ''
        return result


def train_hmm():
    global __hmm
    if not __hmm:
        __hmm = Hmm
    return __hmm


def use_cut():
    hmm_params = Hmm.load_hmm_params()
    return Viterbit(hmm_params)
