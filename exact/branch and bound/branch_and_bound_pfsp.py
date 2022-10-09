# * Zhou Hong
# * hongzhou_ie@163.com
# * Nanjing University of Aeronautics and Astronautics

# from gurobipy import *
import copy
import math

import numpy as np
import pandas as pd
import time
import matplotlib as plt


class Node:
    def __init__(self, job=None, index=None, pa=None):
        self.ub = None
        self.lb = math.inf
        self.pa = pa
        self.of = []
        self.job = job
        # self.index = index
        self.tao = None
        self.EF = None  # 工序的完工时间矩阵
        self.obj = math.inf
        self.series = []
        self.unseries = []
        self.visited = False
        self.update()

    def update(self):  # 初始化更新
        if self.pa is not None:
            self.tao = self.pa.tao + 1
            self.ub = self.pa.ub
        else:
            self.tao = 1
        # 更新已排工件和未排工件集，已排工件集包括自身
        if self.job != None:
            self.series.insert(0, self.job)
            fa = self.pa
            for a in range(self.tao - 2):
                if a != 0:
                    fa = fa.pa
                self.series.insert(0, fa.job)
        self.unseries = copy.copy(unseries)
        for x in self.series:
            self.unseries.remove(x)
        # 根据series和unseries来生成downtime，根据series顺序来对工件进行排列
        # 更新EF矩阵
        if self.job != None:  # 如果本节点不是虚构节点，就更新完工时间表
            self.up_EF()

    def create_off(self, off_list, pa_node):
        for i in range(len(off_list)):
            J = Node(off_list[i], i, pa_node)
            # J.update()
            self.of.append(J)

    def up_ub(self, obj):
        self.ub = obj

    def up_lb(self, obj):
        self.lb = obj

    def up_EF(self):  # 更新完工时间矩阵
        if self.tao == 2:  # 当本工序为除虚构工序外的第一道工序时,此时tao为2
            self.EF = np.ones((pt.shape[0], len(release))) * np.inf
            for i in range(self.EF[:, 0].shape[0]):
                if i == 0:
                    self.EF[:, 0][i] = release[self.job] + pt[:, self.job][i]
                else:
                    self.EF[:, 0][i] = self.EF[:, 0][i - 1] + pt[:, self.job][i]
        else:  # 如不为第一道工序，直接在上一时间矩阵的基础上更新
            self.EF = copy.deepcopy(self.pa.EF)
            # 更新
            for ma in range(pt.shape[0]):
                if ma == 0:
                    self.EF[ma, self.tao - 2] = max(release[self.job], self.EF[ma, self.tao - 3]) \
                                                   + pt[ma, self.job]
                else:  # 同机器上前一道工序完工时间和上一台机器上同工件的上一道工序的完工时间取大
                    self.EF[ma, self.tao - 2] = max(self.EF[ma, self.tao - 3], self.EF[ma - 1, self.tao - 2]) \
                                                   + pt[ma, self.job]


def generate_ub():
    # 根据FCFS规则生成上界
    s = [[0] * 4 for i in range(3)]  # 开始时间
    c = [[0] * 4 for i in range(3)]  # 完成时间
    n = np.argsort(release)

    s[0][n[0]] = release[n[0]]
    c[0][n[0]] = s[0][n[0]] + p_ij[0][n[0]]

    for j in range(1, 4):
        s[0][n[j]] = max(c[0][n[j - 1]], release[n[j]])

    for i in range(1, 3):
        s[i][n[0]] = c[i - 1][n[0]]
        c[i][n[0]] = s[i][n[0]] + p_ij[i][n[0]]

    for i in range(1, 3):
        for j in range(1, 4):
            s[i][n[j]] = max(c[i][n[j - 1]], c[i - 1][n[j]], release[n[j]])
            c[i][n[j]] = s[i][n[j]] + p_ij[i][n[j]]

    ub = c[2][n[3]]  # 求出的上界
    return ub


def cut_1(series_, unseries_, pt_, EF_):
    lb_visit = copy.copy(unseries_)
    for judge_1 in range(len(lb_visit)):  # 判断judge_1是否可以剪枝
        list = copy.copy(unseries_)
        list.pop(judge_1)  # list是unseries中除judge_1外的其他工件
        for j in list:
            EF_left = 0  # 第一台机器上前一个工序_(:з」∠)_的完工时间
            if series_:
                EF_left = EF_[:, series_[-1]][0]
            # 如果释放时间和上一道工序之间可以放其他工件，则剪掉
            if release[unseries_[judge_1]] - EF_left >= release[j] + pt_[0, j]:
                lb_visit.remove(unseries_[judge_1])
                break
    return lb_visit  # 返回可访问的工件列表


def lb_calculation(unseries_, series_, pt, release, EF_):  # 计算此节点的下界，传入未排工件集和已排工件集和此时完工时间矩阵
    lb_m = [0 for i in range(num_m)]  # 用于存储此时机器中的下界，方便比较
    for machine in range(num_m):
        time_record = copy.deepcopy(EF_)  # 进行深拷贝，存储一次循环中的完工时间
        for i in range(len(unseries_)):  # 遍历按照释放时间排序的unseries
            j = unseries_[i]  # 工件编号
            if series_:
                if release[j] <= time_record[machine, i + len(series_) - 1]:  # 如果释放时间小于上一道工序的完工时间
                    time_record[machine, i + len(series_)] = time_record[machine, i + len(series_) - 1] + pt[:, j][
                        machine]  # 上一道工序的完工时间加上本工序的加工时间
                else:
                    time_record[machine, i + len(series_)] = release[j] + pt[:, j][machine]
            else:  # 当属于第一台排序的工件时
                if i == 0:
                    time_record[machine, i + len(series_)] = pt[:, j][machine]
                elif release[j] <= time_record[machine, i + len(series_) - 1]:
                    time_record[machine, i + len(series_)] = time_record[machine, i + len(series_) - 1] + pt[:, j][
                        machine]
                else:
                    time_record[machine, i + len(series_)] = release[j] + pt[:, j][machine]
        if machine != num_m - 1:  # 不是最后一台机器
            lb_m[machine] = time_record[machine, pt.shape[1] - 1] + \
                            min([pt[machine + 1:, :][:, job].sum() for job in unseries_])
        else:
            lb_m[machine] = time_record[machine, pt.shape[1] - 1]
    return max(lb_m)


def branch_and_bound(node):
    global ub
    if node.visited == False:  # 节点的子代未探索完时
        # 剪枝1
        if node.tao != num_jobs + 1:  # 如果不是后两层一层，则剪枝生成子枝
            node.unseries = cut_1(node.series, node.unseries, pt, node.EF)
            # node.update()
            node.create_off(node.unseries, node)
            node.visited = True
            # 下界计算，每次计算完和全局上界对比
            for i in node.of:
                # i.up_EF()
                if i.tao != num_jobs + 1:
                    i.up_lb(lb_calculation(i.unseries, i.series, pt, release, i.EF))
                else:
                    i.up_lb(i.EF[-1][-1])
                # 剪枝2
                if ub < i.lb:  # 下界大于上界剪掉
                    node.of.remove(i)
                elif i.lb < ub:  # 下界小于上界，更新上界
                    ub = i.lb
        else:  # 如果是最后一层，只需要更新，并将此节点记录至最优列表
            # node.update()
            best_list.append(node)
            node.visited = True
    elif node.pa != None:
        if node != node.pa.of[-1]:  # 有右节点，访问右节点
            return branch_and_bound(node.pa.of[node.pa.of.index(node) + 1])
        elif node == node.pa.of[-1]:  # 无右节点，返回至父节点
            if node.tao != num_jobs + 1:
                return branch_and_bound(node.pa)
    else:
        return best_list
    # 最后一层的处理
    if node.tao != num_jobs + 1:  # 如果tao还没到最后一层，则向下探索
        if len(node.of) != 0:
            return branch_and_bound(node.of[0])
        else:
            return branch_and_bound(node.pa)
    elif node.tao == num_jobs + 1:  # tao已经是最后一层，返回至父节点
        return branch_and_bound(node.pa)


num_m = 3  # 机器个数
# 释放时间
release = [5, 1, 0, 9]
# 加工时间
p_ij = [[3, 2, 4, 1],
        [8, 4, 5, 7],
        [4, 5, 6, 3]]
ub = generate_ub()  # 生成上界
num_jobs = len(release)   # 工件数，也是一个可行解的节点数，+1代表添加了一个0工件
pt = np.array(p_ij)
EF = np.ones((pt.shape[0], len(release))) * np.inf
series = []
unseries = np.array(list(range(4)))
unseries = unseries[np.argsort(release)]
unseries = list(unseries)
rootnode = Node()
tao = 1
rootnode.update()  # 更新
rootnode.up_ub(ub)  # 更新上界
rootnode.EF = EF # 更新完工时间列表
best_list = []


time_s = time.time()
branch_and_bound(rootnode)
time_e = time.time()
time_c = time_e - time_s
best_schedule = None
for node in best_list:
    if node.lb == ub:
        best_schedule = node
print("最优解目标值为：%d " % best_schedule.lb)
print("其完工时间列表为：\n", best_schedule.EF)
print("工件顺序依次为：", best_schedule.series)
print("求解耗时：%d 秒" % time_c)