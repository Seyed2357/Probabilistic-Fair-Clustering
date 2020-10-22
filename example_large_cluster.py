import configparser
import sys
import timeit
from pathlib import Path
import numpy as np 
import pandas as pd 
from fair_clustering_large_cluster import fair_clustering_large_cluster
from util.configutil import read_list
from util.utilhelpers import max_Viol_multi_color, x_for_colorBlind, max_RatioViol_multi_color, find_balance_multi_color, max_Viol_Normalized_multi_color

# num_colors: number of colors according to the data set 
num_colors = 7 

# Choose the following 
LowerBound = 7000

# if ml_model_flag = True then p_acc will not be used 
ml_model_flag = True
p_acc = 0.9 

config_file = "config/example_large_cluster_config.ini"
config = configparser.ConfigParser(converters={'list': read_list})
config.read(config_file)


config_str = "census1990_ss_age_7_classes" if len(sys.argv) == 1 else sys.argv[1]


# Read variables
data_dir = config[config_str].get("data_dir")
dataset = config[config_str].get("dataset")
clustering_config_file = config[config_str].get("config_file")
num_cluster = list(map(int, config[config_str].getlist("num_clusters")))
deltas = list(map(float, config[config_str].getlist("deltas")))
max_points = config[config_str].getint("max_points")


 
if type(num_cluster) is list:
    num_cluster = num_cluster[0] 

output = fair_clustering_large_cluster(dataset, clustering_config_file, data_dir, num_cluster, deltas, max_points, LowerBound, p_acc, ml_model_flag)














