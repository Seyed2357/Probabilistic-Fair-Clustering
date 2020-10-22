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

# Choose the list of lower bounds  
LowerBounds = [7000,8000,9000,10000]

# if ml_model_flag = True then p_acc will not be used 
ml_model_flag = True
p_acc = 0.9 

config_file = "config/example_large_cluster_config.ini"
config = configparser.ConfigParser(converters={'list': read_list})
config.read(config_file)

# Create your own entry in `example_config.ini` and change this str to run
# your own trial
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

df = pd.DataFrame(columns=['L','POF','MaxViolFair','MaxViolUnFair','maxRatioViolFairNorm','maxRatioViolUnFairNorm','MaxViol_L_Fair','MaxViol_L_UnFair','MaxViolRatioFair','MaxViolRatioUnFair','Fair Balance','UnFair Balance','Run_Time','NF_Time','NF_prcentTime','ColorBlindCost','FairCost'])
iter_idx = 0 

for L in LowerBounds:
	start_time = timeit.default_timer()
	output = fair_clustering_large_cluster(dataset, clustering_config_file, data_dir, num_cluster, deltas, max_points, L, p_acc, ml_model_flag)
	elapsed_time = timeit.default_timer() - start_time

	#
	fair_cost = output['objective']
	colorBlind_cost = output['unfair_score']
	POF = fair_cost/colorBlind_cost

	#
	x_rounded = output['assignment'] 
	x_color_blind = x_for_colorBlind(output['unfair_assignments'],num_cluster)

	#
	scaling = output['scaling']
	clustering_method = output['clustering_method']

	#
	alpha_dic = output['alpha']
	beta_dic = output['beta']
	( _ , alpha_dic), = alpha_dic.items()
	( _ , beta_dic), = beta_dic.items()

	alpha= np.zeros(num_colors)
	beta= np.zeros(num_colors)

	for k,v in alpha_dic.items():
		alpha[int(k)] = v
	for k,v in beta_dic.items():
		beta[int(k)] = v 


	num_points = sum(x_rounded)

	assert sum(x_rounded)==sum(x_color_blind)

	prob_vecs = output['prob_vecs'] 
	prob_vecs = np.reshape(prob_vecs, (-1,num_colors)) 

	maxViolFair = max_Viol_multi_color(x_rounded,num_colors,prob_vecs,num_cluster,alpha,beta)
	maxViolUnFair = max_Viol_multi_color(x_color_blind,num_colors,prob_vecs,num_cluster,alpha,beta)

	maxViol_L_Fair = maxViolFair/L
	maxViol_L_UnFair = maxViolUnFair/L

	maxRatioViolFair = max_RatioViol_multi_color(x_rounded,num_colors,prob_vecs,num_cluster,alpha,beta)
	maxRatioViolUnFair = max_RatioViol_multi_color(x_color_blind,num_colors,prob_vecs,num_cluster,alpha,beta)


	maxRatioViolFairNorm = max_Viol_Normalized_multi_color(x_rounded,num_colors,prob_vecs,num_cluster,alpha,beta)
	maxRatioViolUnFairNorm = max_Viol_Normalized_multi_color(x_color_blind,num_colors,prob_vecs,num_cluster,alpha,beta)


	proportion_data_set = np.sum(prob_vecs,axis=0)/num_points

	fair_balance = find_balance_multi_color(x_rounded,num_colors,num_cluster,prob_vecs,proportion_data_set)
	unfair_balance = find_balance_multi_color(x_color_blind,num_colors,num_cluster,prob_vecs,proportion_data_set)

	nf_time = output['nf_time'] 
	nf_time_percent = nf_time/elapsed_time

	df.loc[iter_idx] = [L,POF,maxViolFair,maxViolUnFair,maxRatioViolFairNorm,maxRatioViolUnFairNorm,maxViol_L_Fair,maxViol_L_UnFair,maxRatioViolFair,maxRatioViolUnFair,fair_balance,unfair_balance,elapsed_time,nf_time,nf_time_percent,colorBlind_cost,fair_cost]

	iter_idx += 1 





scale_flag = 'normalized' if scaling else 'unnormalized' 
ml_model = 'withMLmodel' if ml_model_flag else 'NoMLmodel'

filename = dataset + '_' + clustering_method + '_' + str(int(num_points)) + '_' + scale_flag + '_' + ml_model 


if ml_model_flag==False:
	if p_acc!=1:
		p_acc_str = 'p' + str(p_acc-int(p_acc))[2:]
		filename = filename + '_' + p_acc_str
	else:
		p_acc_str = str(1)


filename = filename + '.csv'


# do not over-write 
filepath = Path('Results' + '/'+ filename)
while filepath.is_file():
	filename='new' + filename 
	filepath = Path('Results' + '/'+ filename)

df.to_csv('Results' + '/'+ filename, sep=',',index=False)





