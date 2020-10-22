import configparser
import sys
import numpy as np
import pandas as pd 
import timeit
from pathlib import Path



from fair_clustering_2_color import fair_clustering_2_color
from util.configutil import read_list
from util.utilhelpers import max_Viol, x_for_colorBlind, find_balance, find_proprtions_two_color

num_colors = 2 


config_file = "config/example_2_color_config.ini"
config = configparser.ConfigParser(converters={'list': read_list})
config.read(config_file)

# Create your own entry in `example_config.ini` and change this str to run
# your own trial
# bank_binary_marital 
config_str = "bank_binary_marital" if len(sys.argv) == 1 else sys.argv[1]



# Read variables
data_dir = config[config_str].get("data_dir")
dataset = config[config_str].get("dataset")
clustering_config_file = config[config_str].get("config_file")
num_clusters = list(map(int, config[config_str].getlist("num_clusters")))
deltas = list(map(float, config[config_str].getlist("deltas")))
max_points = config[config_str].getint("max_points")
p_acc = float(config[config_str].get("p_acc")) 



output = fair_clustering_2_color(dataset, clustering_config_file, data_dir, num_clusters, deltas, max_points, 0, p_acc)









