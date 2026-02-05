# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 11:03:41 2024

@author: zeyu
"""

import numpy
import pandas as pd
from sub_probelm import sp_func
from mater_probelm import NE_func
from config import *
from dict_pro import sp_path_pro, mp_pro
from PDN_disflow import opf_disflow, Bus_information, Line_information

TN_gragh = pd.read_excel("./5 node transportation network.xlsx", sheet_name="link info", header=0)
OD_info = pd.read_excel("./5 node transportation network.xlsx", sheet_name="OD info", header=0)

link_num = TN_gragh.shape[0]
OD_num = OD_info.shape[0]

origin_node = 1
destination_node = 5

time_set = list(range(10))
charger_slow_loc = [1]
charger_fast_loc = [2, 3, 4]

filepath = r"./configs/default_new.json"
config = get_config(filepath)
config_ = {"link_num": link_num, "charger_fast_loc": charger_fast_loc,
           "charger_slow_loc": charger_slow_loc}
config.update(config_)

node_num =config['node_num']

prices_dict = {}
for i, j in enumerate(charger_fast_loc):
    prices_dict[j] = 0.05*0.8
for i, j in enumerate(charger_slow_loc):
    prices_dict[j] = 0.05

arc_list = []

# 时间上移动
for i in range(link_num):
    for t in time_set:
        index_i = (i, t)
        index_j = (i, t + 1)
        if i == origin_node - 1 and t + 1 < 10:
            arc_list.append((i + 1, i + 1, t, t + 1))
        if i == destination_node - 1 and t + 1 < 10:
            arc_list.append((i + 1, i + 1, t, t + 1))
            
# 空间上移动
for t in time_set:
    for i in range(link_num):
        for j in range(link_num):
            for w in range(link_num):
                if i == TN_gragh['link-i'][w] - 1 and j == TN_gragh['link-j'][w] - 1:
                    arc_list.append((i + 1, j + 1, t, t))
link = arc_list

arc_list = []
for i in range(link_num):
    for j in range(link_num):
        for w in range(link_num):
            if i == TN_gragh['link-i'][w] - 1 and j == TN_gragh['link-j'][w] - 1:
                arc_list.append((i + 1, j + 1))
link_2 = arc_list

def cal_NE(s_charging_P):
    # 读入 free-flow travel time 和 capacity
    t0_dict = {}
    cap_dict = {}
    da_dict = {}
    va_dict = {}
    vn_dict = {}
    F_t = {}
    path_3 = {}
    path_2 = {}
    path_num_3 = {}
    path_num_2 = {}
    F_cost = {}

    for w in range(OD_num):
        F_t[w] = {}
        path_num_3[w] = 0
        path_3[w] = []
        for t in time_set:
            path_2[w, t] = []
            path_num_2[w, t] = 0
        for p in range(100):
            for p1 in range(100):
                for t in time_set:
                    F_t[w][p, p1, t] = 0

    key_list = []
    for i in range(link_num):
        key = int(TN_gragh['link-i'][i]), int(TN_gragh['link-j'][i])
        key_list.append(key)
        t0_dict[key] = TN_gragh['free-flow travel time'][i]
        cap_dict[key] = TN_gragh['capacity'][i]
        da_dict[key] = 2.5 * t0_dict[key]

    for i in range(link_num):
        for t in time_set:
            key = int(TN_gragh['link-i'][i]), int(TN_gragh['link-j'][i]), t
            va_dict[key] = 0

    for i in range(node_num):
        for t in time_set:
            vn_dict[i + 1, t] = 0
    
    # sp中定义优化变量时的索引key值
    node_set = list(range(1, node_num + 1))
    lines = pd.read_excel(r"./5 node transportation network.xlsx", sheet_name="OD info", header=0)
    w_list = []
    for i in range(OD_num):
        key = (int(lines['link-i'][i]), int(lines['link-j'][i]))
        w_list.append(key)

    travel_demand = {0: 5000}
    config_dict = {'t0_dict': t0_dict, 'cap_dict': cap_dict, 'da_dict': da_dict, 'node': node_set, 'time_set': time_set,
                   'link': link, 'link_2': link_2,
                   'path_num_2': path_num_2, 'F_t': F_t, 'path_num_3': path_num_3, 'path_3': path_3, 'path_2': path_2,
                   'w_list': w_list, 'F_cost': F_cost, 'travel_demand': travel_demand}

    P_set_fast = {}
    P_set_fast_epl = {}
    P_set_slow = {}
    P_set_slow_epl = {}

    for w in range(OD_num):
        for t in time_set:
            for p in range(10):
                for n in range(node_num):
                    P_set_fast[w, t, p, n] = 0
                    P_set_fast_epl[w, t, p, n] = 0
                    P_set_slow[w, t, p, n] = 0
                    P_set_slow_epl[w, t, p, n] = 0
                    
    config_dict['P_set_fast'] = P_set_fast
    config_dict['P_set_fast_epl'] = P_set_fast_epl
    config_dict['P_set_slow'] = P_set_slow
    config_dict['P_set_slow_epl'] = P_set_slow_epl

    for w_index in range(OD_num):
        w = w_list[w_index]
        sp_reult = sp_func(w, prices_dict, vn_dict, va_dict, config, config_dict, s_charging_P)
        sp_path_pro(w_index, sp_reult, config, config_dict)

    charging_schedul = [[sp_reult['sp_e_slow'], sp_reult['sp_e_fast']]]

    for i in range(10):
        
        print('迭代次数', i+1)

        flag_break = 1

        mp_result = NE_func(config, config_dict)
    
        vn_dict, va_dict = mp_pro(mp_result, config, config_dict, vn_dict, va_dict)
        
        travel_cost_main = mp_result['dual_mu']
        
        for w in range(OD_num):
            
            sp_reult = sp_func(w_list[w], prices_dict, vn_dict, va_dict, config, config_dict,
                               s_charging_P)
            
            aguimented_path = sp_path_pro(w, sp_reult, config, config_dict)
            
            print('sp', sp_reult['travel_cost'], 'mp', travel_cost_main[w])
            
            charging_schedul.append([sp_reult['sp_e_slow'], sp_reult['sp_e_fast']])
            
            
            if sp_reult['travel_cost'] + travel_cost_main[w]*0.01 <= travel_cost_main[w]:
                flag_break = 0

        if flag_break:
            break
    # if flag_break == 0:
    #     print('not optimal solution')

    print('aguimented_path', aguimented_path)
    s_charging_dict = {(j, t):0 for j in charger_slow_loc for t in time_set}
    s_charging_num = {(j, t):0 for j in charger_slow_loc for t in time_set}
    f_charging_dict = {(j, t):0 for j in charger_fast_loc for t in time_set}

    for w in range(OD_num):
        for i in range(len(charging_schedul)-1):
            for j in charger_slow_loc:
                for t in time_set:
                    s_charging_dict[j, t] += mp_result['F_pw_solution'][w, i] * charging_schedul[i][0][j, t] / 1000
            for j in charger_fast_loc:
                for t in time_set:
                    f_charging_dict[j, t] += mp_result['F_pw_solution'][w, i] * charging_schedul[i][1][j, t] / 1000

    for w in range(OD_num):
        for i in range(len(charging_schedul)-1):
            for j in charger_slow_loc:
                for t in time_set:
                    if charging_schedul[i][0][j, t] >= 0.001:
                        s_charging_num[j, t] += mp_result['F_pw_solution'][w, i]
    
    result = {'s_charging_num':s_charging_num,'f_charging_dict':f_charging_dict,
              's_charging_dict':s_charging_dict, "vn_dict":vn_dict, "va_dict":va_dict,
              'F_pw_solution':mp_result['F_pw_solution']}
    return result


if __name__ == '__main__':
    
    s_charging_P = {}
    for i in charger_slow_loc:
        for t in time_set:
            s_charging_P[i,t] = 2.5
    
    result_NE = cal_NE(s_charging_P)
    
    s_charging_dict_op = result_NE['s_charging_dict']
    f_charging_dict_op = result_NE['f_charging_dict']
    F_pw_solution_op = result_NE['F_pw_solution']    
    
# if __name__ == '__main__':
    
#     Bus_set = list(range(Bus_information.shape[0]))
#     Line_set = list(range(Line_information.shape[0]))
#     time_set = list(range(10))
    
#     s_charging_P_list =[]
#     P_ch_s_ini_list = []
    
#     s_charging_P = {}
#     for i in charger_slow_loc:
#         for t in time_set:
#             s_charging_P[i,t] = 2.5
    
#     k = 0
    
#     while True:
        
#         print('第{}次迭代'.format(k))

#         result_NE = cal_NE(s_charging_P)
        
#         s_charging_num = result_NE['s_charging_num']
        
#         P_ch_s_ini = {}
#         for i in charger_slow_loc:
#             for t in time_set:
#                 P_ch_s_ini[t] = result_NE['s_charging_dict'][(i, t)]
        
#         P_ch_s_ini_list.append(P_ch_s_ini)
        
#         result_opf = opf_disflow(P_ch_s_ini)
        
#         s_charging_P_list.append(s_charging_P)
        
#         s_charging_P = {}
        
#         for i in charger_slow_loc:
#             for t in time_set:
#                 if s_charging_num[(i,t)]>0:
#                     s_charging_P[(i,t)] = result_opf['P_ch_s_dict'][t]*1000/s_charging_num[(i,t)]
#                     if s_charging_P[(i,t)]>5.0:
#                         s_charging_P[(i,t)] = 5.0
#                 else:
#                     s_charging_P[(i,t)] = 2.5
        
#         k += 1
        
#         if k>=2 and s_charging_P_list[-1] == s_charging_P_list[-2]:
#             print(k)
#             break
#         if k>= 20:
#             break
    
# In[]

# s_charging_P1 = s_charging_P_list[-1]
# print(s_charging_P1)
# result_NE1 = cal_NE(s_charging_P1)
# print(result_NE1['s_charging_dict'])


# s_charging_P2 = s_charging_P_list[-2]
# print(s_charging_P2)
# result_NE2 = cal_NE(s_charging_P2)
# print(result_NE2['s_charging_dict'])












