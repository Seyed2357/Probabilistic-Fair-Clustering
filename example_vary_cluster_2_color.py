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

# k0: is the first cluster size 
k0= 10

# kend: is the last cluster size 
kend= 20



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
deltas = list(map(float, config[config_str].getlist("deltas")))
max_points = config[config_str].getint("max_points")
p_acc = float(config[config_str].get("p_acc")) 



# ready up for the loop 
clusters = [ k+k0 for k in list(range(kend-k0+1))]
df = pd.DataFrame(columns=['num_clusters','POF','MaxViolFair','MaxViolUnFair','Fair Balance','UnFair Balance','Run_Time','ColorBlindCost','FairCost'])
iter_idx = 0 


#delta = 0.01


for cluster in clusters:
    start_time = timeit.default_timer()
    output = fair_clustering_2_color(dataset, clustering_config_file, data_dir, cluster, deltas, max_points, 0, p_acc)
    elapsed_time = timeit.default_timer() - start_time


    fair_cost = output['objective']
    colorBlind_cost = output['unfair_score']
    POF = fair_cost/colorBlind_cost

    x_rounded = output['assignment'] 
    x_color_blind = x_for_colorBlind(output['unfair_assignments'],cluster)


    scaling = output['scaling']
    clustering_method = output['clustering_method']
    #alpha = 0.744615
    #beta = 0.603138
    alpha = output['alpha']
    beta = output['beta']
    prob_vals = output['prob_values'] 

    num_points = sum(x_rounded)
    assert sum(x_rounded)==sum(x_color_blind)

    ( _ , alpha), = alpha.items()
    ( _ , beta), = beta.items()
    ( _ , prob_vals), = prob_vals.items()

    maxViolFair = max_Viol(x_rounded,num_colors,prob_vals, cluster,alpha,beta)
    maxViolUnFair = max_Viol(x_color_blind,num_colors,prob_vals, cluster,alpha,beta)


    proportion_data_set = sum(prob_vals)/num_points 
    fair_balance = find_balance(x_rounded,num_colors, cluster,prob_vals,proportion_data_set)
    unfair_balance = find_balance(x_color_blind,num_colors, cluster,prob_vals,proportion_data_set)


    _, props , sizes= find_proprtions_two_color(x_rounded,num_colors,prob_vals,cluster)


    df.loc[iter_idx] = [cluster,POF,maxViolFair,maxViolUnFair,fair_balance,unfair_balance,elapsed_time,colorBlind_cost,fair_cost]

    iter_idx += 1 



scale_flag = 'normalized' if scaling else 'unnormalized' 
filename = dataset + '_' + clustering_method + '_' + str(int(num_points)) + '_' + scale_flag  
p_acc_str = 'p' + str(p_acc-int(p_acc))[2:]
filename = filename + '_' + p_acc_str
filename = filename + '.csv'



# do not over-write 
filepath = Path('Results' + '/'+ filename)
while filepath.is_file():
    filename='new' + filename 
    filepath = Path('Results' + '/'+ filename)

df.to_csv('Results' + '/'+ filename, sep=',',index=False)




