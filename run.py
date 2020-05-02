import os
import sys
sys.path.append("/data/chenzhuo/drift/java/execmoa")

import re
import math
import subprocess
import shlex
import string
import pandas as pd
import simpleExperiments as se
import moa_command_vars as mcv
from multiprocessing import Process, Queue
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# 评估指标
evaluations = ['classifications correct (percent)', 'tree size (nodes)']

# 数据流
generators = {
    'LED_g':
        '-i 1000000 -f 1000 -s (ConceptDriftStream -s (generators.LEDGeneratorDrift -d 1)   -d (ConceptDriftStream -s (generators.LEDGeneratorDrift -d 3) -d (ConceptDriftStream -s (generators.LEDGeneratorDrift -d 5)  -d (generators.LEDGeneratorDrift -d 7) -w 50000 -p 250000 ) -w 50000 -p 250000 ) -w 50000 -p 250000)',

    'covtypeNorm': #581011
        '-i 10000000 -f 1000 -s (ArffFileStream -f /data/chenzhuo/drift/java/datasets/covtypeNorm/covtypeNorm.arff -c -1)',
    
    'kdd99': #4898431
        '-i 10000000 -f 10000 -s (ArffFileStream -f /data/chenzhuo/drift/java/datasets/kddcup99/kddcup99.arff -c -1)',

    'SEA_5':
        '-i 5000000 -f 10000 -s (generators.SEAGenerator -b)',
    
    'AGR_a1':
          '-i 10000000 -f 10000 -s (ConceptDriftStream -s (generators.AgrawalGenerator -f 1 -b) -d (ConceptDriftStream -s (generators.AgrawalGenerator -f 2 -b) -d (ConceptDriftStream -s (generators.AgrawalGenerator -b)   -d (generators.AgrawalGenerator -f 4 -b) -w 500 -p 2500000) -w 500 -p 2500000 ) -w 500 -p 2500000)',
    
    'RBF':
        '-i 2000000 -f 10000 -s (generators.RandomRBFGenerator -c 15 -a 200 -n 100) ',
    
    'RTG':
        '-i 10000000 -f 10000 -s (generators.RandomTreeGenerator -c 25 -o 200 -u 200)',
    
    'mnist8m': #6000000
        '-i 10000000 -f 10000 -s (ArffFileStream -f /data/chenzhuo/drift/java/datasets/mnist8m/mnist8m.arff -c 1)',

}

# 输出文件
OUTPUT_DIR = "../final425"

# 评估器
evaluators = [ r"EvaluatePrequential"]

# 学习器
learners = {
    #'CGD': r'-l (trees.HoeffdingTree -a CGD -o '+  OUTPUT_DIR + ')',
    'IMAC': r'-l (trees.HoeffdingTree -i 1 -a INC  -o '+  OUTPUT_DIR + ')',
    'OSM': r'-l (trees.HoeffdingTree  -a OSM -o '+  OUTPUT_DIR + ')',
    'VFDT': r'-l (trees.HoeffdingTree  -a None -o '+  OUTPUT_DIR + ')',
}

def file_to_dataframe(some_file):
    return pd.read_csv(some_file, index_col=False, header=0, skiprows=0)

def read_df_from_folder(folder, column):
    cur_df = pd.DataFrame([])
    files = sorted([file for file in os.listdir(folder) if file.endswith('csv')])
    
    for filename in files:
        file_df = file_to_dataframe(folder+'/'+filename)
        cur_df[str(filename)] = file_df[column]

    return cur_df

def runtime_dict_from_folder(folder):
    runtimes = {}
    files = sorted([file for file in os.listdir(folder) if file.endswith('csv')])
    for filename in files:
        file_df = file_to_dataframe(folder+'/'+filename)
        #file_df['evaluation time (cpu seconds)'].replace('?', 0, inplace=True)
        #file_df['evaluation time (cpu seconds)'].replace('-Infinity', -2147483648, inplace=True)
        runtimes[filename] = file_df['evaluation time (cpu seconds)'].iloc[-1]

    return runtimes

def plot_df(output_dir, evaluations):
    runtime_dict = runtime_dict_from_folder(output_dir) # 先收集时间信息
    info_dfs = []
    
    for target in evaluations:
        cur_df = read_df_from_folder(output_dir, target)
        #cur_df.replace('?', 0.0, inplace=True)
        #cur_df.replace('-Infinity', -2147483648.0, inplace=True)
        new_col_names = ['']*len(cur_df.columns)

        for i, col in enumerate(cur_df.columns):
            cur_df[col] = cur_df[col].astype(float)
            new_col_names[i] = (col.split('.')[0] + " | T:" + ("%.2f s"%runtime_dict[col]) + " | E: " + ("%.4f"%cur_df[col].mean()))
        cur_df.columns = new_col_names
        info_dfs.append(cur_df)
    
    for i in range(len(info_dfs)):
        ax = info_dfs[i].iloc[::10, :].plot()
        #ax = info_dfs[i].plot()
        ax.set_xlabel('Instances (x 1,0000)')
        ax.set_ylabel(evaluations[i])
        fig = ax.get_figure()
        fig.savefig(output_dir + "/" + evaluations[i])
        #fig.close()
        #fig.show()

def runexp(learners, generators, evaluators, evaluations):
    
    for key, generator in generators.items():
        output_dir = OUTPUT_DIR + "/" + str(key)
        genes = {}
        genes[key] = generator
        
        experiments = se.CompositeExperiment.make_experiments(mcv.MOA_STUMP, evaluators, learners, genes)
        processes = se.CompositeExperiment.make_running_processes(experiments, output_dir)
        se.Utils.wait_for_processes(processes)
        
        plot_df(output_dir, evaluations)
'''

def runexp(learners, generators, evaluators, evaluations):
    
    for key, generator in generators.items():
        output_dir = OUTPUT_DIR + "/" + str(key)
        #genes = {}
        #genes[key] = generator
        
        for evaluator in evaluators:
            for learner_name, learner in learners.items():
                experiment = se.Experiment(mcv.MOA_STUMP, evaluator, learner, generator, learner_name, key)
                #experiments = se.CompositeExperiment.make_experiments(mcv.MOA_STUMP, [evaluator], learner, genes)
                processes = se.CompositeExperiment.make_running_processes([experiment], output_dir)
                se.Utils.wait_for_processes(processes)
        
        plot_df(output_dir, evaluations)
'''
        
def work():
    runexp(learners, generators, evaluators, evaluations)


Process(target=work).start()