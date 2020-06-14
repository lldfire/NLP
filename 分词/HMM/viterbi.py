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
    path_v = [{}]
    path_max = {}

    # 计算初始时的状态概率
    for state in states:
        path_v[0][state] = pi_pro[state] * B_pro[state].get(obs[0], 0)
        path_max[state] = [state]

    print(path_max)
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
    print(path_v)
    print(path_max[last_state])


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
    obs1 = ['正常', '冷', '头晕', '头晕', '冷', '冷', '正常', '头晕']
    obs2 = [0, 1, 2, 2, 1, 1, 0, 2]

    viterbit_dict(obs1, states, pi_pro, A_pro, B_pro)
    print(viterbit_mat(obs2, states_mat, pi_pro_mat, A_pro_mat, B_pro_mat))
    # print(help(np.array(states)))
    # print(pd.DataFrame(A_pro).to_numpy())
