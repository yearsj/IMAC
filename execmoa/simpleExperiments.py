
import os, subprocess, shlex, shutil
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pylab
import numpy as np

import re
import utilities
import math
import moa_command_vars as mcv
from textwrap import wrap
from collections import OrderedDict

class Experiment:

  def __init__(self, stump, e, l, g, l_n, g_n):
    if "-o" in l:
      l = l[:-1] + "/" + g_n +")"
    self.cmd = " ".join([stump, "moa.DoTask",  e, l, g])
    self.name = l_n
        
  @staticmethod 
  def make_running_process(exp, output_file):
    
    args = shlex.split(exp.cmd)
    process = subprocess.Popen(args, stdout=open(output_file, "w+"), close_fds=True)
    return process


class CompositeExperiment:

  @staticmethod
  def make_experiments(stump, evaluators, learners, generators):

    experiments = []

    for evaluator in evaluators:  
      for learner_name, learner in learners.items():
        for generator_name, generator in generators.items():
          experiments.append(Experiment(stump, evaluator, learner, generator, learner_name, generator_name))

    return experiments

  @staticmethod
  def make_experiments(stump, evaluators, learners, generators):

    experiments = []

    for evaluator in evaluators:  
      for learner_name, learner in learners.items():
        for generator_name, generator in generators.items():
          experiments.append(Experiment(stump, evaluator, learner, generator, learner_name, generator_name))

    return experiments

  @staticmethod
  def make_running_processes(experiments, output_dir):

    #os.chdir(mcv.MOA_DIR)
    #utilities.remove_folder(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    processes = []
    output_files = []

    for exp in experiments:
      output_file = output_dir + '/' + exp.name + '.csv'
      process = Experiment.make_running_process(exp, output_file)
      processes.append(process)

    return processes

  @staticmethod
  def SimpleSeededGenBuilder(gen_string, randomSeed=None):

    # if random seed is not none, just substitute any -r options with the correct seed
    # the -r options must be clearly visible... 
    # imagine the amount of refactoring needed every time new options are added... that's too
    # much complexity for a piece of code custom-built to work with MOA.

    #print("====" + str(gen_string))
    gen_cmd = " -s \"\"\"(" + re.sub("-r [0-9]+", "-r "+ str(randomSeed)+ " ", str(gen_string)) + " )\"\"\""

    return Generator(gen_cmd)

class Utils: 

  @staticmethod
  def file_to_dataframe(some_file):
    return pd.read_csv(some_file, index_col=False, header=0, skiprows=0)

  @staticmethod
  def dataframe_to_file(some_dataframe, output_csv):
    return some_dataframe.to_csv(some_file, output_csv)

  @staticmethod
  def wait_for_processes(processes):
    exit_codes = [p.wait() for p in processes] #waits for all processes to terminate

  @staticmethod
  def error_df_from_folder(folder):
    error_df = pd.DataFrame([])  
    files = sorted(os.listdir(folder))
    for filename in files:
      file_df = Utils.file_to_dataframe(folder+'/'+filename)
      error_df[str(filename)] = (100.0 - file_df['classifications correct (percent)']) / 100.0

    return error_df

  @staticmethod
  def runtime_dict_from_folder(folder):
    runtimes = {}
    files = sorted(os.listdir(folder))
    for filename in files:
      file_df = Utils.file_to_dataframe(folder+'/'+filename)
      runtimes[filename] = file_df['evaluation time (cpu seconds)'].iloc[-1]

    return runtimes

  @staticmethod
  def split_df_from_folder(folder):
    split_df = pd.DataFrame([])  
    files = sorted(os.listdir(folder))
    for filename in files:
      file_df = Utils.file_to_dataframe(folder+'/'+filename)

      # Only mark actual splits as 1 and discard the rest of the split counts
      splitArray = file_df.loc[:,'splits'].values.tolist()
      i = 0
      while i < len(splitArray)-1:
        #print(str(i+1) + " " + str(splitArray[i+1]) + "\n")
        diff = math.floor(splitArray[i+1]) - math.floor(splitArray[i])
        if(diff > 0):
          splitArray[i+1] = (-1)*diff
          i = i+2
        else:
          i=i+1
      for i in range(len(splitArray)):
        if(splitArray[i] > 0):
          splitArray[i] = 0
        else:
          splitArray[i] = (-1) * splitArray[i]
      split_df[str(filename)] = splitArray

    return split_df

