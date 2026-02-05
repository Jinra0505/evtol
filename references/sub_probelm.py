# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 11:03:41 2024

@author: zeyu
"""

from gurobipy import *
import numpy as np
import pandas as pd

price_info = pd.read_excel("./5 node transportation network.xlsx", sheet_name="price", header=0)

def sp_func(w, prices_dict, vn_dict, va_dict, config, config_dict, s_charging_P):
    node_num = config["node_num"]
    link_num = config["link_num"]
    charger_fast_loc = config["charger_fast_loc"]
    charger_slow_loc = config["charger_slow_loc"]
    w_head = config["w_head"]
    L_max = config["L_max"]
    L0 = eval(config["L0"])
    K = config["K"]
    M = config["M"]
    b0 = config["b0"]
    omega = config["omega"] / 60
    
    time_set = list(range(10))

    

    t0_dict = config_dict['t0_dict']
    cap_dict = config_dict['cap_dict']
    da_dict = config_dict['da_dict']
    link = config_dict['link']
    node = config_dict['node']
    time_set = config_dict['time_set']
    link_2 = config_dict['link_2']
    
    
    b_n_fast = np.zeros((node_num,))
    for i in charger_fast_loc:
        b_n_fast[i - 1] = 100
    
    b_n_slow = {(i,t):s_charging_P[i,t] for i in charger_slow_loc for t in time_set}

    # 形成Δ矩阵 node-link incidence matrix
    delta_list = {}
    for i in node:
        for t in time_set:
            delta_list[i,t] = {}
            for j1, j2, t1, t2 in link:
                delta_list[i, t][j1, j2, t1, t2] = 0

    for i, j, t1, t2 in link:
        delta_list[i, t1][i, j, t1, t2] = 1
        delta_list[j, t2][i, j, t1, t2] = -1

    # 根据当前O-D pair形成E_nw
    E_nw = np.zeros((node_num, len(time_set)))
    for i in range(node_num):
        for t in time_set:
            if i+1 == w[0] and t == 0:
                E_nw[i, t] = 1
            elif i+1 == w[1] and t == len(time_set)-1:
                E_nw[i, t] = -1

    ta_dict = {}  # travel time for each link, 可由给定的va通过严格递增的旅行时间函数计算得到
    
    # ta_dict 作为x_aw的objective coefficient, 可以作为model.addVars中的obj参数传入
    for i, j in link_2:
        for t in time_set:
            ta_dict[i, j, t] = t0_dict[i, j] * (1 + 0.15 * math.pow(va_dict[i, j, t] / cap_dict[i, j], 4))


    tn_dict = {}
    for i in charger_fast_loc:
        for t in time_set:
            tn_dict[i, t] = 5 * (1 + 0.15 * math.pow(vn_dict[i, t] / 5000, 4))

    # Create optimization model
    m = Model('SP')

    # Create variables
    X_aw = m.addVars(link, name="X_aw", vtype=GRB.BINARY)
    ro_aw = m.addVars(link, lb=-GRB.INFINITY, name="ro_aw")
    L_nw = m.addVars(node, time_set, name="L_nw")
    F_nw_fast = m.addVars(node, time_set, name="F_nw_fast")
    F_nw_slow = m.addVars(node, time_set, lb=-1000, name="F_nw_slow")
    epl_fast = m.addVars(node,time_set, name="epl_fast", vtype=GRB.BINARY)
    epl_slow = m.addVars(node, time_set, name="epl_slow", vtype=GRB.BINARY)
    

    # add constraints
    # flow balance constraints X_aw delta_list
    m.addConstrs((quicksum(delta_list[n+1, t][i, j, t1, t2] * X_aw[i, j, t1, t2] for i, j, t1, t2 in link) == E_nw[n, t]
                  for n in range(node_num) for t in time_set), "flow_balance")

    # conservation of electricity for each utilized link(电量守恒)
    m.addConstrs(
        (L_nw[j, t] - L_nw[i, t] + w_head * da_dict[i, j] - F_nw_fast[j, t] == ro_aw[i, j, t, t] for i, j in link_2 for t in time_set),
        "spatial movement electricity balance")

    m.addConstrs(
        (L_nw[i, t+1] - L_nw[i, t] - F_nw_slow[i, t] == ro_aw[i, i, t, t+1] for i in charger_slow_loc for t in time_set if t+1<10),
        "temporal movement electricity balance")

    m.addConstrs(
        (-K * (1 - X_aw[i, j, t1, t2]) <= ro_aw[i, j, t1, t2] for i, j, t1, t2 in link), "ro_aw constraints1")

    m.addConstrs(
        (ro_aw[i, j, t1, t2] <= K * (1 - X_aw[i, j, t1, t2]) for i, j, t1, t2 in link), "ro_aw constraints2")

    # trip requist constraints(行车安全约束，保证在utilized link上不会耗尽电量)
    m.addConstrs(
        (-M * (1 - X_aw[i, j ,t1, t2]) + 0.1 * L_max <= L_nw[i, t1] - w_head * da_dict[i, j] for i, j, t1, t2 in link if t1==t2), "trip request")

    # recharge constraints
    m.addConstrs(
        (0 <= F_nw_fast[i+1, t] for i in range(node_num) for t in time_set), "fast recharge constraints1")
    
    m.addConstrs(
        (F_nw_fast[i+1, t] <= b_n_fast[i] for i in range(node_num) for t in time_set), "fast recharge constraints2")
    
    m.addConstrs(
        (F_nw_fast[i, t] / M <= epl_fast[i, t] for i in node for t in time_set), "fast recharge constraints3")
   
    m.addConstrs(
        (F_nw_slow[i, t] <= epl_slow[i, t] * b_n_slow[i, t] for i in charger_slow_loc for t in time_set), "slow recharge constraints3")
    
    m.addConstrs(
        (F_nw_slow[i, t] >= -epl_slow[i, t] * b_n_slow[i, t] for i in charger_slow_loc for t in time_set), "slow recharge constraints3")
    
    m.addConstrs(
        (F_nw_slow[i, t] == 0 for i in node if i not in charger_slow_loc for t in time_set), "slow recharge constraints3")

    m.addConstrs(
        (epl_fast[i, t] <= F_nw_fast[i, t] * M for i in node for t in time_set), "recharge constraints4")
   
    # battery constraints
    m.addConstrs(
        (0.1 * L_max <= L_nw[i, t] for i in node for t in time_set), "battery constraints")

    m.addConstrs(
        (L_nw[i, t] <= L_max for i in node for t in time_set), "battery constraints")

    # initial state
    m.addConstr((L_nw[w[0], 0] == L0), "initial state")
    
    
    # early arrive constraint
    m.addConstr(9 - quicksum(t * X_aw[i, j, t, t] for t in time_set for i, j in link_2 if i != w[1]
                          if delta_list[w[1], t][i, j, t, t] == -1) <= 2)

    obj1 = LinExpr(0.0)
    obj2 = LinExpr(0.0)

    for i, j in link_2:
        for t in time_set:
            obj1 += omega * X_aw[i, j, t, t] * ta_dict[i, j, t]
    
    for t in time_set:
        for i in charger_fast_loc:
            obj2 += F_nw_fast[i, t] * prices_dict[i] * price_info['price'][t] + omega * epl_fast[i, t] * tn_dict[i, t]
        for i in charger_slow_loc:
            obj2 += F_nw_slow[i, t] * prices_dict[i] * price_info['price'][t]

    # 提前到达惩罚
    obj2 += 30 * (9 - quicksum(t * X_aw[i, j, t, t] for t in time_set for i, j in link_2 if i != w[1]
                          if delta_list[w[1], t][i, j, t, t] == -1)) * omega

    obj = obj1 + obj2

    # Compute optimal solution
    m.Params.OutputFlag = 0
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    if m.status == GRB.Status.OPTIMAL:
        
        sp_w = dict(m.getAttr('x', X_aw))
        sp_e_fast = dict(m.getAttr('x', F_nw_fast))
        sp_e_slow = dict(m.getAttr('x', F_nw_slow))
    
        sp_epl_fast = dict(m.getAttr('x', epl_fast))
        sp_epl_slow = dict(m.getAttr('x', epl_slow))
        sp_L_nw = dict(m.getAttr('x', L_nw))
        travel_cost = obj.getValue()
        travel_cost2 = obj2.getValue()

        result = {'sp_w': sp_w, "sp_e_fast":sp_e_fast, "sp_e_slow":sp_e_slow, 'sp_L_nw':sp_L_nw,
                  "sp_epl_fast":sp_epl_fast, 'sp_epl_slow':sp_epl_slow, 'delta_list':delta_list,
                  'travel_cost':travel_cost, 'travel_cost2':travel_cost2}
        
        return result
    else:
        print('sp not_at_optimal')
        return {}



if __name__ == '__main__':
    print('hello world')