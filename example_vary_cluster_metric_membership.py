import configparser
import sys
import pandas as pd 
import timeit
from pathlib import Path


from fair_clustering_metric_membership import fair_clustering_metric_membership
from util.configutil import read_list
from util.utilhelpers import max_Viol, x_for_colorBlind, find_balance


num_colors = 2

# k0: is the first cluster size 
k0= 15

# kend: is the last cluster size 
kend= 20




config_file = "config/example_metric_membership_config.ini"
config = configparser.ConfigParser(converters={'list': read_list})
config.read(config_file)

# Create your own entry in `example_config.ini` and change this str to run
# your own trial
config_str = "adult_age" if len(sys.argv) == 1 else sys.argv[1]



# Read variables
data_dir = config[config_str].get("data_dir")
dataset = config[config_str].get("dataset")
clustering_config_file = config[config_str].get("config_file")
deltas = list(map(float, config[config_str].getlist("deltas")))
max_points = config[config_str].getint("max_points")



# ready up for the loop 
clusters = [ k+k0 for k in list(range(kend-k0+1))]
df = pd.DataFrame(columns=['num_clusters','POF','MaxViolFairNormalized','MaxViolUnFairNormalized','MaxViolFair','MaxViolUnFair','Fair Balance','UnFair Balance','Run_Time','ColorBlindCost','FairCost','R_max'])
iter_idx = 0 





for cluster in clusters:
    start_time = timeit.default_timer()

    output = fair_clustering_metric_membership(dataset, clustering_config_file, data_dir, cluster, deltas, max_points, 0)
    elapsed_time = timeit.default_timer() - start_time


    fair_cost = output['objective']
    colorBlind_cost = output['unfair_score']
    POF = fair_cost/colorBlind_cost

    x_rounded = output['assignment'] 
    x_color_blind = x_for_colorBlind(output['unfair_assignments'],cluster)


    scaling = output['scaling']
    clustering_method = output['clustering_method']

    alpha = output['alpha']
    beta = output['beta']
    prob_vals = output['prob_values'] 

    num_points = sum(x_rounded)
    assert sum(x_rounded)==sum(x_color_blind)

    ( _ , alpha), = alpha.items()
    ( _ , beta), = beta.items()
    ( _ , prob_vals), = prob_vals.items()

    R_max = output['R_max']

    maxViolFair = max_Viol(x_rounded,num_colors,prob_vals, cluster,alpha,beta)
    maxViolUnFair = max_Viol(x_color_blind,num_colors,prob_vals, cluster,alpha,beta)

    maxViolFairNormalized = maxViolFair/R_max
    maxViolUnFairNormalized = maxViolUnFair/R_max

    proportion_data_set = sum(prob_vals)/num_points 
    fair_balance = find_balance(x_rounded,num_colors, cluster,prob_vals,proportion_data_set)
    unfair_balance = find_balance(x_color_blind,num_colors, cluster,prob_vals,proportion_data_set)


    df.loc[iter_idx] = [cluster,POF,maxViolFairNormalized,maxViolUnFairNormalized,maxViolFair,maxViolUnFair,fair_balance,unfair_balance,elapsed_time,colorBlind_cost,fair_cost,R_max]

    iter_idx += 1 




scale_flag = 'normalized' if scaling else 'unnormalized' 
filename = dataset + '_' + clustering_method + '_' + str(int(num_points)) + '_' + scale_flag + '_' + str(R_max) 
filename = filename + '.csv'

print(df)

# do not over-write 
filepath = Path('Results' + '/'+ filename)
while filepath.is_file():
    filename='new' + filename 
    filepath = Path('Results' + '/'+ filename)

df.to_csv('Results' + '/'+ filename, sep=',',index=False)