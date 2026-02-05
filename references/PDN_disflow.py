# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 11:03:41 2024

@author: zeyu
"""

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import *


Bus_information = pd.read_excel(r".\PDN_info.xlsx", sheet_name="bus", header=0)  # 节点电压信息
Line_information = pd.read_excel(r".\PDN_info.xlsx", sheet_name="branch", header=0)  # 支路信息
Price_e = pd.read_excel(r".\PDN_info.xlsx", sheet_name="price", header=0)  # 实时电价
factor = pd.read_excel(r".\PDN_info.xlsx", sheet_name="Tan", header=0)  # 功率因数
PV = pd.read_excel(r".\PDN_info.xlsx", sheet_name="PV", header=0)  # 实时光伏有功出力
PVi = pd.read_excel(r".\PDN_info.xlsx", sheet_name="PVi", header=0)  # 光伏接入点信息
PV_bus = pd.read_excel(r".\PDN_info.xlsx", sheet_name="PVi", header=0)  # 光伏接入节点编号

active = pd.read_excel(r".\PDN_info.xlsx", sheet_name="active", header=0)      #节点有功
reactive = pd.read_excel(r".\PDN_info.xlsx", sheet_name="reactive", header=0)  #节点无功1



def opf_disflow(P_ch_s_ini):
    
    #定义gurobi模型
    opf_down = gp.Model()
    
    Bus_set = list(range(Bus_information.shape[0]))
    Line_set = list(range(Line_information.shape[0]))
    time_set = list(range(10))
    
    V_bus = opf_down.addVars(Bus_set, time_set, name='V_bus')  # 节点电压幅值平方
    
    I_branch = opf_down.addVars(Line_set, time_set, lb=0, name='I_branch')  # 支路电流平方
    
    Pij = opf_down.addVars(Line_set, time_set, lb=0, ub=GRB.INFINITY, name='Pij')  # 支路有功功率流
    Qij = opf_down.addVars(Line_set, time_set, lb=0, ub=GRB.INFINITY, name='Qij')  # 支路无功功率流
    
    Pi_bus = opf_down.addVars(Bus_set, time_set, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Pi_bus')  # 节点注入有功功率
    Qi_bus = opf_down.addVars(Bus_set, time_set, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Qi_bus')  # 节点注入无功功率
    
    P_g = opf_down.addVars(Bus_set, time_set, name='P_g')  # 发电机有功出力
    Q_g = opf_down.addVars(Bus_set, time_set, name='Q_g')  # 发电机无功出力
    
    P_cut = opf_down.addVars(Bus_set,time_set, name='P_cut')
    Q_cut = opf_down.addVars(Bus_set,time_set, name='Q_cut')
    
    
    P_ch_s = opf_down.addVars(time_set, lb=-GRB.INFINITY, name='P_ch_s') # 充电负荷慢充功率
    
    P_sub = opf_down.addVars(time_set, lb=0, name='P_sub')  # 根节点上级电网有功
    Q_sub = opf_down.addVars(time_set, lb=0, name='Q_sub')  # 根节点上级电网无功
    
    V_fbus = opf_down.addVars(Line_set,time_set, name='V_fbus')  # 某支路首端电压平方
    V_tbus = opf_down.addVars(Line_set,time_set, name='V_tbus')  # 某支路末端电压平方

    obj = opf_down.addVar(name='obj')  # 目标函数

    Bus_set = list(range(Bus_information.shape[0]))
    Line_set = list(range(Line_information.shape[0]))

    # 支路阻抗的平方
    Z = {}
    for l in Line_set:
        Z[l] = Line_information['r'][l] ** 2 + Line_information['x'][l] ** 2
        
    a = 0.05
    b = 0.38
    d = 10
    opf_down.addConstr(obj >= quicksum(a * P_g[i,t] ** 2 + b * P_g[i,t] + d * P_cut[i,t] for t in time_set for i in Bus_set) +
                       quicksum(Price_e['price'][t] * P_sub[t] for t in time_set))
    

    # 节点电压约束和支路电流约束
    opf_down.addConstrs(V_bus[i, t] <= Bus_information['Vmax'][i] ** 2 for i in Bus_set for t in time_set)
    opf_down.addConstrs(V_bus[i, t] >= Bus_information['Vmin'][i] ** 2 for i in Bus_set for t in time_set)
        
    opf_down.addConstrs(V_fbus[l,t] <= Bus_information['Vmax'][l] ** 2 for l in Line_set for t in time_set)
    opf_down.addConstrs(V_fbus[l,t] >= Bus_information['Vmin'][l] ** 2 for l in Line_set for t in time_set)
    opf_down.addConstrs(V_tbus[l,t] <= Bus_information['Vmax'][l] ** 2 for l in Line_set for t in time_set) 
    opf_down.addConstrs(V_tbus[l,t] >= Bus_information['Vmin'][l] ** 2 for l in Line_set for t in time_set)
    opf_down.addConstrs(I_branch[l,t] <= Line_information['Imax'][l] ** 2 for l in Line_set for t in time_set)

    # 支路首、末端电压与支路的联系
    opf_down.addConstrs(V_fbus[l, t] == V_bus[i, t] for t in time_set 
                        for l in Line_set for i in Bus_set if Line_information['fbus'][l] - 1 == i)

    opf_down.addConstrs(V_tbus[l, t] == V_bus[i, t] for t in time_set 
                                    for l in Line_set for i in Bus_set if Line_information['tbus'][l] - 1 == i) 

    # 支路首末端电压关系
    opf_down.addConstrs(V_fbus[l, t] * I_branch[l, t] >= Pij[l, t] ** 2 + Qij[l, t] ** 2 for l in Line_set for t in time_set)  
    opf_down.addConstrs(V_tbus[l, t] == V_fbus[l, t] - 2 * (Pij[l, t] * Line_information['r'][l]
                                                      + Qij[l, t] * Line_information['x'][l]) + I_branch[l, t] * Z[l] for l in Line_set for t in time_set) 


    # 潮流方程
    opf_down.addConstrs(Pi_bus[i, t] ==  quicksum(Pij[l, t] - I_branch[l, t] * Line_information['r'][l] 
                                               for l in Line_set if Line_information['tbus'][l] - 1 == i)
                                                    - quicksum(Pij[l,t] for l in Line_set if Line_information['fbus'][l] - 1 == i)
                                                                    for i in Bus_set for t in time_set)
    
    opf_down.addConstrs(Qi_bus[i, t] ==  quicksum(Qij[l, t] - I_branch[l, t] * Line_information['x'][l] 
                                               for l in Line_set if Line_information['tbus'][l] - 1 == i)
                                                    - quicksum(Qij[l, t] for l in Line_set if Line_information['fbus'][l] - 1 == i)
                                                                    for i in Bus_set for t in time_set)

    # 充电负荷动态调节
    opf_down.addConstrs(P_ch_s[t] <= P_ch_s_ini[t]*1.2 for t in time_set)
    opf_down.addConstrs(P_ch_s[t] >= P_ch_s_ini[t]*0.8 for t in time_set)

    
    opf_down.addConstrs(quicksum(P_ch_s[t] for t in time_set) == quicksum(P_ch_s_ini[t] for t in time_set) for i in Bus_set)
    
    
    # 节点有功平衡
    bus_slcak = {i:0 for i in Bus_set}
    bus_slcak[0] = 1
    bus_generation = {i:0 for i in Bus_set}
    bus_generation[12] = 1
    bus_generation[31] = 1
    s_charge_loc = {i:0 for i in Bus_set}
    s_charge_loc[26] = 1
    
    opf_down.addConstrs(Pi_bus[i,t] + P_sub[t]*bus_slcak[i] + P_g[i,t]*bus_generation[i]
                        == active[t][i] + P_ch_s[t]*s_charge_loc[i] - P_cut[i,t] for i in Bus_set for t in time_set)
            
    # 节点无功平衡约束
    opf_down.addConstrs(Qi_bus[i,t] + Q_sub[t]*bus_slcak[i] + Q_g[i,t]*bus_generation[i]
                        == reactive[t][i] + P_ch_s[t]*factor['tan'][i]*s_charge_loc[i] - Q_cut[i,t] for i in Bus_set for t in time_set)
    
    opf_down.addConstrs(active[t][i] - P_cut[i,t] >= 0 for i in Bus_set for t in time_set)
    opf_down.addConstrs(reactive[t][i] - Q_cut[i,t] >= 0 for i in Bus_set for t in time_set)

    opf_down.setObjective(obj, gp.GRB.MINIMIZE)
    opf_down.setParam('OutputFlag', 0)
    opf_down.optimize()

    if opf_down.status == GRB.Status.OPTIMAL:
        
        P_cut_dict = dict(opf_down.getAttr('x', P_cut))
        P_g_dict = dict(opf_down.getAttr('x', P_g))
        P_sub_dict = dict(opf_down.getAttr('x', P_sub))
        P_ch_s_dict = dict(opf_down.getAttr('x', P_ch_s))
        
        opf_result = {"P_cut_dict":P_cut_dict, 'P_g_dict':P_g_dict, 'P_sub_dict':P_sub_dict, 'operation_cost': obj.x,
                      'P_ch_s_dict':P_ch_s_dict}
        
        return opf_result
    else:
        print('disflow not optimal')

if __name__ == '__main__':
    
    Bus_set = list(range(Bus_information.shape[0]))
    Line_set = list(range(Line_information.shape[0]))
    time_set = list(range(10))
    
    P_ch_s_ini = {t:0 for t in time_set}
    
    i = 32
    for t in range(10):
        P_ch_s_ini[t] = s_charging_dict[(1, t)]
    
    opf_result = opf_disflow(P_ch_s_ini)
    
    print('before', P_ch_s_ini)
    
    print('after', opf_result['P_ch_s_dict'])
    
    
    s_charge_P = {}
    for t in range(10):
        if s_charging_num[(1,t)]>0:
            s_charge_P[t] = opf_result['P_ch_s_dict'][t]*1000/s_charging_num[(1,t)]
            if s_charge_P[t]>5.0:
                s_charge_P[t] = 5.0
        else:
            s_charge_P[t] = 2.5
