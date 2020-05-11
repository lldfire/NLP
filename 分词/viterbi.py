# 实现：https://www.zhihu.com/question/20136144第二个回答的代码
import numpy as np
import pandas as pd


states = ['健康', '发烧']     # 隐藏状态
observe = ['正常', '冷', '头晕']    # 观测值
pi_pro = {states[0]: 0.6, states[1]: 0.4}     # 初始状态分布矩阵
# 状态转移矩阵
A_pro = {
    states[0]: {states[0]: 0.7, states[1]: 0.3},
    states[1]: {states[0]: 0.4, states[1]: 0.6}
}
# 观测值发射矩阵
B_pro = {
    states[0]: {observe[0]: 0.5, observe[1]: 0.4, observe[2]: 0.1},
    states[1]: {observe[0]: 0.1, observe[1]: 0.3, observe[2]: 0.6},
}

# 已知观测序列：正常、冷、头晕，求隐藏的健康状态
# 类比分词，已知字符的排列数据，求每个字符的隐藏状态

# 计算第一天健康情况下的正常的概率
# p_1_health = pi_pro[state[0]] * B_pro[state[0]][observe[0]]
# p_1_hot = pi_pro[state[1]] * B_pro[state[1]][observe[0]]
# print(p_1_health)
# print(p_1_hot)

# # 计算第二天的概率值
# p_2_health = max([p_1_health * A_pro[state[0]][state[0]], p_1_hot *\
#     A_pro[state[0]][state[0]]]) * B_pro[state[0]][observe[1]]
# p_2_hot = max([p_1_health * A_pro[state[0]][state[1]], p_1_hot *\
#     A_pro[state[1]][state[1]]]) * B_pro[state[1]][observe[1]]
# print(p_2_health)
# print(p_2_hot)


def viterbit_dict(obs, states, pi_pro, A_pro, B_pro):
    # init path: path[s] represents the path ends with s
    path = {s: [] for s in states}
    curr_pro = {}
    # 开始时刻时，各个状态下出现该观测值的概率
    for s in states:
        curr_pro[s] = pi_pro[s] * B_pro[s][obs[0]]
    # print(curr_pro)
    for i in range(1, len(obs)):
        last_pro = curr_pro     # 上一个状态的概率分布
        curr_pro = {}
        for curr_state in states:
            # 上一个状态到当前状态的概率值最大的路径和状态
            # 上一个状态发生的概率 * 从上一个状态转移为当前状态的概率 * 
            # 当前状态下的观测值的概率
            max_pro, last_sta = max(((last_pro[last_state] * A_pro[last_state][curr_state] * B_pro[curr_state][obs[i]], last_state)
                                     for last_state in states))
            curr_pro[curr_state] = max_pro
            path[curr_state].append(last_sta)
            # print(curr_pro)
            # print(path)

    # find the final largest probability
    max_pro = -1
    max_path = None
    # print(path)
    # print(curr_pro)
    for s in states:
        path[s].append(s)
        # print(path)
        # 比较最后一个状态的可能概率值
        if curr_pro[s] > max_pro:
            max_path = path[s]
            max_pro = curr_pro[s]
        # print('%s: %s' % (curr_pro[s], path[s]))
    print(max_pro)
    return max_path


# 矩阵的形势表示HMM的元素
states_mat = np.array([0, 1])
observe_mat = np.array([0, 1, 2])
pi_pro_mat = np.array([0.6, 0.4])
A_pro_mat = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])
B_pro_mat = np.array([
    [0.5, 0.4, 0.1],
    [0.1, 0.3, 0.6]
])


def viterbit_mat(obs, states, pi_pro, A_pro, B_pro):
    # 构建一个n * m的矩阵,n：状态的长度，m：观测值长度
    F = np.zeros((states.shape[0], len(obs)))
    # 计算初始状态下的概率矩阵
    F[:, 0] = pi_pro_mat * B_pro[:, obs[0]]
    # print(F)
    # 计算之后的状态概率
    for l in range(1, F.shape[1]):
        max_list = []
        for s in states:
            max_list.append((F[:, l - 1] * A_pro[:, s]).max())
        F[:, l] = np.array(max_list) * B_pro[:, obs[l]]
    
    # 根据F得出最终的状态序列
    print(F)
    F.argmax(axis=0)
    return [['健康', '发烧'][idx] for idx in F.argmax(axis=0)]


if __name__ == '__main__':
    # obs1 = ['正常', '冷', '头晕', '头晕', '冷', '冷', '正常']
    # obs2 = [0, 1, 2, 2, 1, 1, 0]

    # print(viterbit_dict(obs1, states, pi_pro, A_pro, B_pro))
    # print(viterbit_mat(obs2, states_mat, pi_pro_mat, A_pro_mat, B_pro_mat))
    # print(help(np.array(states)))
    print(pd.DataFrame(A_pro).to_numpy())