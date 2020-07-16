#! /usr/bin/env python
#-*- coding: utf-8 -*-
 
#####################################################
# Copyright (c) 2019 USTC, Inc. All Rights Reserved
#####################################################
# File:    mlsa.py
# Author:  cb
# Date:    2019/01/02 10:52:18
# Brief:
#####################################################

import sys
import os  


def diff_list(models):
    diff = []
    same = {}
    for model in models:
        for key in model.keys():
            if key == 'result' or key == 'dir' or key in diff:
                continue
            if key not in diff and key not in same:
                same[key] = model[key]
            if key in same:
                if same[key] == model[key]:
                    continue
                else:
                    diff.append(key)
                    same.pop(key)
    return diff
            

def arrange(models, compare, i):
    if i == len(compare):
        assert len(models) == 1
        return models[0]['result']
    key = compare[i]
    s_models = {}
    for model in models:
        value = model[key]
        if value not in s_models:
            s_models[value] = [model]
        else:
            s_models[value].append(model)
    sifter = {}
    for value in s_models.keys():
        sifter[key+'|'+value] = arrange(s_models[value], compare, i+1)
    return sifter
        

def put(sifter, log):
    if isinstance(sifter, dict):
        for key in sifter.keys():
            put(sifter[key], log+('' if len(log) == 0 else '_')+key)
    elif isinstance(sifter, list):
        i = 0
        for l in sifter:
            i += 1
            data = [i] + [log] + l['n_runs'] + [l['avgacc']] + [l['stdev']]
            print('\t'.join(map(str, data)))


          
result_dir = sys.argv[1]
dataset_name = sys.argv[2]
model_name = sys.argv[3]
for _, d, _ in os.walk(result_dir):
    dirs = d
    break
models = []
for d in dirs:
    eles = d.split('_')
    info = {}
    if eles[0] != 'training':
        continue
    for i in range(int(len(eles)/2)):
        info[eles[2*i+1]] = eles[2*i+2]
    info['dir'] = d
    if info['cfm'] == model_name and info['dt'] == dataset_name:
        log_dir = result_dir + ('' if result_dir[-1] == '/' else '/') + d + '/log_files'
        for _, _, f in os.walk(log_dir):
            files = f
            break
        result = []
        for f in files:
            try:
                one = {}
                file_name = log_dir + '/' + f
                lines = open(file_name, 'r').readlines()
                one['name'] = f
                one['n_runs'] = [float(x) for x in lines[-2].strip()[16:].split(',')]
                one['avgacc'] = float(lines[-1][lines[-1].find('accuracy is')+12:].split(',')[0])
                one['stdev'] = float(lines[-1].strip()[lines[-1].find('variance is')+12:])
                result.append(one)
            except:
                continue
        info['result'] = result
        models.append(info)
compare = diff_list(models)
n = len(compare)
sifter = arrange(models, compare, 0)
put(sifter, '')
        
        
















# vim: set expandtab ts=4 sw=4 sts=4 tw=100
