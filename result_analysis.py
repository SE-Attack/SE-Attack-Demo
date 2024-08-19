"""
This code implements tool functions for analyzing attack results from log files.
"""

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import matplotlib.font_manager as fm


def get_exp_result(path, num=1000):
    with open(path, 'r') as recordfile:
        exp_result = recordfile.read().strip().split('\n')
    exp_result_np = np.zeros((num,), dtype=int)
    for img_record in exp_result:
        temp = img_record.split(' ')
        imid = int(temp[1][:-1])
        queries = int(temp[3][:-1])
        exp_result_np[imid] = queries
    return exp_result_np

def get_exp_result_2(path):
    with open(path, 'r') as recordfile:
        exp_result = recordfile.read().strip().split('\n')
    succ_idx = list()
    queries_list = list()
    for img_record in exp_result:
        temp = img_record.split(' ')
        imid = int(temp[1][:-1])
        queries = int(temp[3][:-1])

        if imid < 250:
            succ_idx.append(imid)
            queries_list.append(queries)
    queries_list = np.array(queries_list)
    avg_queries = np.mean(queries_list)
    std_queries = np.std(queries_list)
    report = f"ASR: {len(succ_idx)}\t\t avg queries: {avg_queries:>.1f}~{std_queries:>.1f}, "
    print(report)
    return succ_idx

def result_analysis(exp_np):
    succ_exp_np = exp_np[exp_np>0]
    avg_queries = np.mean(succ_exp_np)
    std_queries = np.std(succ_exp_np)
    total_success = len(succ_exp_np)
    success_rate = total_success/len(exp_np)

    report = f"total_success: {total_success}; success_rate: {success_rate:>.4f}, avg queries: {avg_queries:>.1f}~{std_queries:>.1f}, "
    return report

def queries_succrate_analysis(exp_np):
    exp_succ_np = exp_np[exp_np != 0]
    query_item = np.unique(exp_succ_np)
    query_total_succ = []
    for query in query_item:
        temp_total_succ = len(exp_succ_np[exp_succ_np<=query])
        query_total_succ.append(temp_total_succ)
    query_total_succ = np.array(query_total_succ)
    query_total_succ = query_total_succ/float(len(exp_np))
    return list(zip(query_item, query_total_succ))
    
def exp_analysis(path):
    run_name = os.path.split(path)[-1]
    target_file = f"{path}/{run_name}.txt"
    untarget_file = f"{path}/{run_name}.txt"
    
    target_result_np = get_exp_result(target_file)
    untarget_result_np = get_exp_result(untarget_file)

    target_report = result_analysis(target_result_np)
    untarget_report = result_analysis(untarget_result_np)

    target_query_curve = queries_succrate_analysis(target_result_np)
    untarget_query_curve = queries_succrate_analysis(untarget_result_np)

    print("target. {}\nuntarget. {}".format(target_report, untarget_report))
    return target_query_curve, untarget_query_curve

def exp_analysis_2(path, run_name):
    #run_name = os.path.split(path)[-1]
    target_file = f"{path}/{run_name}.txt"
    untarget_file = f"{path}/{run_name}_pretend.txt"
    
    target_result_np = get_exp_result(target_file)
    untarget_result_np = get_exp_result(untarget_file)

    target_report = result_analysis(target_result_np)
    untarget_report = result_analysis(untarget_result_np)

    target_query_curve = queries_succrate_analysis(target_result_np)
    untarget_query_curve = queries_succrate_analysis(untarget_result_np)

    print("target. {}\nuntarget. {}".format(target_report, untarget_report))
    return target_query_curve, untarget_query_curve

def get_gcv_result(dir_path):
    result_dir = "{}/gcv_attack_info_ours.txt".format(dir_path)
    with open(result_dir, 'r') as recordfile:
        exp_result = recordfile.read().strip().split('\n')
    exp_result_np = np.zeros((100, ), dtype=int)
    for img_record in exp_result:
        temp = img_record.split(' ')
        imid = int(temp[1][:-1])
        queries = int(temp[-1])
        exp_result_np[imid] = queries
    return exp_result_np

def gcv_result_analysis(exp_np):
    avg_queries = np.mean(exp_np[exp_np!=0])
    std_queries = np.std(exp_np[exp_np!=0])
    success_rate = float(len(exp_np[exp_np!=0])/float(len(exp_np)))

    falid_im_ids = [i for i,q in enumerate(exp_np) if q>0]
    total_success = len(exp_np[exp_np!=0])
    report = f"gcv untarget. total_success: {total_success}; success_rate: {success_rate:>.4f}, avg queries: {avg_queries:>.1f}~{std_queries:>.1f}, "
    print(report)
    return avg_queries, success_rate, falid_im_ids
