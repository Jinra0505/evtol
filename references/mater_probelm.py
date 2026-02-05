# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 11:03:41 2024

@author: zeyu
"""


from gurobipy import *

def NE_func(config, config_dict):

    OD_num = config["OD_num"]
    omega = config["omega"]
    EVCS_capa = config["EVCS_capa"]
    charger_fast_loc = config["charger_fast_loc"]
    charger_slow_loc = config["charger_slow_loc"]

    link = config_dict['link'] # 三维广义路径
    link_2 = config_dict['link_2'] # 二维路径
    cap_dict = config_dict['cap_dict']
    w_list = config_dict['w_list']
    node = config_dict['node']
    t0_dict = config_dict['t0_dict']

    path_num_3 = config_dict['path_num_3']
    path_num_2 = config_dict['path_num_2']
    F_cost = config_dict['F_cost']
    F_t = config_dict['F_t']
    time_set = config_dict['time_set']
    P_set = config_dict['path_2'] # OD对路径集合
    P_set_fast_epl = config_dict['P_set_fast_epl'] # OD对 时间 路径 充电站
    travel_demand = config_dict['travel_demand'] # 路径集合
    
    cap_dict = config_dict['cap_dict']

    # Create a new model
    m = Model('NE')
    # Create variables
    key_3 = []
    for w in range(OD_num):
        for i in range(path_num_3[w]):
            key_3.append((w, i))
            
    key_2 = []
    for w in range(OD_num):
        for t in time_set:
            for i in range(path_num_2[w, t]):
                key_2.append((w, i, t))
                
    def calculate_slope(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        if x2 - x1 == 0:
            raise ValueError("这两个点在垂直线上，斜率为无穷大")
        slope = (y2 - y1) / (x2 - x1)
        return slope

    va_point = {}  # 分段线性化 点的位置
    va_fd_len = [int(0.1 * cap_dict[l]  - 1) for l in link_2]

    for l1,l in enumerate(link_2):
        my_list = []
        for j in range(0, int(10 * cap_dict[l]), va_fd_len[l1]):
            my_list.append([j, t0_dict[l] * (1 + 0.15 * (j / (cap_dict[l])) ** 4)])
        va_point[l] = my_list

    va_slope = {}  # 分段线性化 斜率
    for l in link_2:
        my_list = va_point[l]
        slope_list = []
        for i in range(len(my_list) - 1):
            point1 = my_list[i]
            point2 = my_list[i + 1]
            slope = calculate_slope(point1, point2)
            slope_list.append(slope)
        va_slope[l] = slope_list
    
    va_fd_list = []
    for i, j in enumerate(link_2):
        for w in range(len(va_slope[j])):
            va_fd_list.append((i, w))
            
    cap_EVCS = {2:5000, 3:5000, 4:5000}
    vn_point = {}  # 分段线性化 点的位置
    vn_fd_len = [int(0.05 * cap_EVCS[l]  - 1) for l in charger_fast_loc]
    
    for l1,l in enumerate(charger_fast_loc):
        my_list = []
        for j in range(0, int(10 * cap_EVCS[l]), vn_fd_len[l1]):
            my_list.append([j, 5 * (1 + 0.15 * (j / (cap_EVCS[l])) ** 4)])
        vn_point[l] = my_list

    vn_slope = {}  # 分段线性化 斜率
    for l in charger_fast_loc:
        my_list = vn_point[l]
        slope_list = []
        for i in range(len(my_list) - 1):
            point1 = my_list[i]
            point2 = my_list[i + 1]
            slope = calculate_slope(point1, point2)
            slope_list.append(slope)
        vn_slope[l] = slope_list
    
    vn_fd_list = []
    for i, j in enumerate(charger_fast_loc):
        for w in range(len(vn_slope[j])):
            vn_fd_list.append((i, w))
    
    OD_set = list(range(OD_num))

    F_pw = m.addVars(key_3, name='F_pw')  # 广义路径（时间和空间都有）
    C_pw = m.addVars(key_3, name='C_pw')  # 广义路径成本（时间和空间都有）
    B_pw = m.addVars(key_3, vtype=GRB.BINARY, name='C_pw')  # 广义路径（时间和空间都有）
    lameda = m.addVars(OD_set, name='C_pw')  # 广义路径（时间和空间都有）
    
    f_pw = m.addVars(key_2, name='f_pw')  # 代表第w个ODpair的第几个可行路径对应的流量
    va = m.addVars(link_2, time_set, name='va')
    ta = m.addVars(link_2, time_set, name='ta')
    va_fd = m.addVars(va_fd_list, time_set, vtype=GRB.BINARY, name='va_fd')
    va_relax = m.addVars(va_fd_list, time_set, name='va_relax')
    
    vn = m.addVars(charger_fast_loc, time_set, name='vn')
    tn = m.addVars(charger_fast_loc, time_set, name='tn')
    vn_fd = m.addVars(vn_fd_list, time_set, vtype=GRB.BINARY, name='vn_fd')
    vn_relax = m.addVars(vn_fd_list, time_set, name='vn_relax')
    
    
    path_2 = [[] for t in time_set]
    for w in range(OD_num):
        for t in time_set:
            if path_num_2[w,t] == 0:
                a = l
            else:
                a = path_num_2[w,t]
            for p in range(a):   
                path_2[t].append([w, p])
    
    # add constraints
    fpw_dict = {}
    for t in time_set:
        for l1, l2 in link_2:
            fpw_dict[l1, l2, t] = {}
            for w, p in path_2[t]:
                fpw_dict[l1, l2, t][w, p, t] = 0
                

    for w in range(OD_num):
        for t in time_set:
            for j in range(path_num_2[w, t]):
                path = P_set[w, t][j]
                for p, q in path:
                    fpw_dict[p, q, t][w, j, t] = 1

    fpw_node_dict = {}
    for t in time_set:
        for n in node:
            fpw_node_dict[n, t] = {}

    for t in time_set:
        for n in node:
            for w, t1, i in key_2:
                fpw_node_dict[n, t][w, t1, i] = 0

    for w in range(OD_num):
        for t in time_set:
            for j in range(path_num_2[w, t]):
                path = P_set[w, t][j]
                for p, q in path:
                    if p in charger_fast_loc and P_set_fast_epl[w, t, j, p] == 1:
                        fpw_node_dict[p, t][w, j, t] = 1
                    if q in charger_fast_loc and P_set_fast_epl[w, t, j, q] == 1:
                        fpw_node_dict[q, t][w, j, t] = 1
                        
    Fpw_dict = {}  
    for t in time_set:
        for l1, l2 in link_2:
            Fpw_dict[l1, l2, t] = {}
            for w, p in key_3:
                Fpw_dict[l1, l2, t][w, p] = sum(fpw_dict[l1, l2, t][w, p1, t]*\
                                                   F_t[w][p, p1, t] for p1 in range(path_num_2[w, t]))
    
    Fpw_node_dict = {}  
    for t in time_set:
        for n in node:
            Fpw_node_dict[n, t] = {}
            for w, p in key_3:
                Fpw_node_dict[n, t][w, p] = sum(fpw_node_dict[n, t][w, p1, t]*\
                                                   F_t[w][p, p1, t] for p1 in range(path_num_2[w, t]))
                    
    m.addConstrs((quicksum(F_pw[w, p] for p in range(path_num_3[w])) == travel_demand[w]
                  for w in OD_set), 'demand balance')

    m.addConstrs(
        (f_pw[w, p1, t] == quicksum(F_pw[w, p] * F_t[w][p, p1, t] for p in range(path_num_3[w]))
        for w in range(OD_num) for t in time_set for p1 in range(path_num_2[w, t])), "dimensional reduction")

    m.addConstrs(
        (va[p, q, t] == f_pw.prod(fpw_dict[p, q, t]) for p, q in link_2 for t in time_set), "va_definition")

    m.addConstrs(
        (vn[i, t] == f_pw.prod(fpw_node_dict[i, t]) for i in charger_fast_loc for t in time_set), "vn_definition")

    # 路段流量 分段线性化约束
    m.addConstrs(quicksum(va_relax[w,j, t] * va_slope[l][j] + va_fd[w, j, t] * va_point[l][j][1]
                          for w, j in va_fd_list if link_2[w] == l) == ta[l[0],l[1],t] 
                 for l in link_2 for t in time_set)

    m.addConstrs(va[l[0],l[1],t] == quicksum(va_fd[w, j, t] * va_point[l][j][0] + va_relax[w, j, t]
                 for w, j in va_fd_list if link_2[w] == l) for l in link_2 for t in time_set)

    m.addConstrs(va_relax[w, j, t] <= va_fd[w, j, t] * va_fd_len[w]
                 for w, j in va_fd_list for t in time_set)
    
    m.addConstrs(va_fd.sum(w1, "*", t) == 1
                  for w1, w2 in enumerate(link_2) for t in time_set)

    # 充电站车流 分段线性化约束
    m.addConstrs(quicksum(vn_relax[w, j, t] * vn_slope[l][j] + vn_fd[w, j, t] * vn_point[l][j][1]
                          for w, j in vn_fd_list if charger_fast_loc[w] == l) == tn[l, t]
                  for l in charger_fast_loc for t in time_set)

    m.addConstrs(vn[l, t] == quicksum(vn_fd[w, j, t] * vn_point[l][j][0] + vn_relax[w, j, t]
                  for w, j in vn_fd_list if charger_fast_loc[w] == l) 
                  for l in charger_fast_loc for t in time_set)

    m.addConstrs(vn_relax[w, j, t] <= vn_fd[w, j, t] * vn_fd_len[w]
                  for w, j in vn_fd_list for t in time_set)
    
    m.addConstrs(vn_fd.sum(w1, "*", t) == 1
                  for w1, w2 in enumerate(charger_fast_loc) for t in time_set)

    # 广义路径成本计算
    m.addConstrs(C_pw[w, p] ==  10/60*quicksum(ta[l[0], l[1], t]*Fpw_dict[l[0], l[1], t][w,p] for l in link_2 for t in time_set)+\
                  10/60*quicksum(tn[n,t]*Fpw_node_dict[n,t][w,p] for n in charger_fast_loc for t in time_set)+\
                      F_cost[w,p]
                  for w, p in key_3)

    # 交通均衡互补松弛约束
    m.addConstrs(C_pw[w, p] - lameda[w] >= 0 for w, p in key_3)
    
    m.addConstrs(C_pw[w, p] - lameda[w] <= B_pw[w,p]*1e5 for w, p in key_3)
    
    m.addConstrs(F_pw[w, p] <= (1-B_pw[w,p]) * 1e5 for w, p in key_3)
    
    obj = m.addVar(name='obj', ub=GRB.INFINITY)
    
    m.addConstr(obj == quicksum(lameda[w] * travel_demand[w] for w in OD_set))

    m.setObjective(obj, GRB.MINIMIZE)
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = 600
    m.optimize()

    if m.status == GRB.Status.OPTIMAL:
        # print('mp is optimal')
        va_solution = dict(m.getAttr('x', va))
        vn_solution = dict(m.getAttr('x', vn))
        f_pw_solution = dict(m.getAttr('x', f_pw))
        F_pw_solution = dict(m.getAttr('x', F_pw))
        dual_mu = dict(m.getAttr('x', lameda))
        total_cost = m.objVal
        
        # print('total_cost', total_cost)

        # my_Constr = {}
        # dual_mu = []
        # for w in range(OD_num):
        #     my_Constr[w] = m.getConstrByName(name='demand balance[{}]'.format(w))
        #     dual_mu.append(my_Constr[w].Pi)
        
    else:
        va_solution = {}
        vn_solution = {}
        dual_mu = {}
        p_ch = {}
        demand_solution = {}
        print('mp not at optimal')

    result_list = {"va_solution":va_solution, 'vn_solution':vn_solution, 'dual_mu':dual_mu,
                   'f_pw_solution':f_pw_solution, 'F_pw_solution':F_pw_solution, 'fpw_dict':fpw_dict}
    return result_list



