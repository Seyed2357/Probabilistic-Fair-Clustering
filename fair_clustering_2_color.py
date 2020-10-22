import configparser
import time
import pickle 
from collections import defaultdict
from functools import partial
import numpy as np
import pandas as pd
from cplex_fair_assignment_lp_solver_2_color import fair_partial_assignment_2_color
from util.clusteringutil import (clean_data, read_data, scale_data,
                                 subsample_data, take_by_key,
                                 vanilla_clustering, write_fairness_trial)
from util.configutil import read_list
from util.probutil import form_class_prob_vector, sample_colors, create_prob_vecs, perturb_2_color




# This function takes a dataset and performs a fair clustering on it.
# Arguments:
#   dataset (str) : dataset to use
#   config_file (str) : config file to use (will be read by ConfigParser)
#   data_dir (str) : path to write output
#   num_clusters (int) : number of clusters to use
#   deltas (list[float]) : delta to use to tune alpha, beta for each color
#   max_points (int ; default = 0) : if the number of points in the dataset 
#       exceeds this number, the dataset will be subsampled to this amount.
# Output:
#   None (Writes to file in `data_dir`)  
def fair_clustering_2_color(dataset, config_file, data_dir, num_clusters, deltas, max_points, L=0, p_acc=1.0):
    # NOTE: thos code works for 2 colors 
    num_colors = 2

    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(config_file)

    # Read data in from a given csv_file found in config
    df = read_data(config, dataset)
    # Subsample data if needed
    if max_points and len(df) > max_points:
       # NOTE: comment the block and second and unccomment the second block. changed to exclude randomization effect
       #rows = [0,1,2,3,4,5,20,21,23,50,126,134,135]
       #df = df.iloc[rows,:]
       #df = df.reset_index()
       

       df = df.head(max_points)
       # below if you wish to shuffle 
       # df= df.sample( frac=1, random_state=1).reset_index(drop=True)





    # Clean the data (bucketize text data)
    df, _ = clean_data(df, config, dataset)


    # variable_of_interest (list[str]) : variables that we would like to collect statistics for
    variable_of_interest = config[dataset].getlist("fairness_variable")
    
    # NOTE: this code only handles one color per vertex 
    assert len(variable_of_interest) == 1 

    # Assign each data point to a color, based on config file
    # attributes (dict[str -> defaultdict[int -> list[int]]]) : holds indices of points for each color class
    # color_flag (dict[str -> list[int]]) : holds map from point to color class it belongs to (reverse of `attributes`)
    attributes, color_flag, prob_vecs, prob_vals, prob_vals_thresh, prob_thresh = {}, {}, {}, {}, {}, {}
    for variable in variable_of_interest:
        colors = defaultdict(list)
        this_color_flag = [0] * len(df)
        
        condition_str = variable + "_conditions"
        bucket_conditions = config[dataset].getlist(condition_str)

        # For each row, if the row passes the bucket condition, 
        # then the row is added to that color class
        for i, row in df.iterrows():
            for bucket_idx, bucket in enumerate(bucket_conditions):
                if eval(bucket)(row[variable]):
                    colors[bucket_idx].append(i)  # add the point to the list of its colors 
                    this_color_flag[i] = bucket_idx  # record the color for this given point 


        attributes[variable] = colors     
        color_flag[variable] = this_color_flag

        # NOT: generate probabilities according to the perturbation descired in section 5.2  
        prob_vals[variable] = [perturb_2_color(color,p_acc) for color in this_color_flag]



    # representation (dict[str -> dict[int -> float]]) : representation of each color compared to the whole dataset
    representation ={} 
    for var in variable_of_interest:
        representation[var] = sum(prob_vals[var])/len(df)

   

    ( _ , fair_vals), = representation.items()


    # drop uneeded columns 
    selected_columns = config[dataset].getlist("columns")
    df = df[[col for col in selected_columns]]

    # Scale data if desired
    scaling = config["DEFAULT"].getboolean("scaling")
    if scaling:
        df = scale_data(df)




    # Cluster the data -- using the objective specified by clustering_method
    clustering_method = config["DEFAULT"]["clustering_method"]


    t1 = time.monotonic()
    # NOTE: initial_score is the value of the objective at the solution 
    # NOTE: This is where the color-blind algorithm is ran  
    if type(num_clusters) is list:
        num_clusters = num_clusters[0] 

    initial_score, pred, cluster_centers = vanilla_clustering(df, num_clusters, clustering_method)
    t2 = time.monotonic()
    cluster_time = t2 - t1
    print("Clustering time: {}".format(cluster_time))
    

    # sizes (list[int]) : sizes of clusters
    sizes = [0 for _ in range(num_clusters)]
    for p in pred:
        sizes[p] += 1




    # fairness_vars (list[str]) : Variables to perform fairness balancing on
    fairness_vars = config[dataset].getlist("fairness_variable")

    # NOTE: here is where you set the upper and lower bounds 
    # NOTE: accross all different values within the same attribute you have the same multipliers up and down 
    for delta in deltas:
        alpha, beta = {}, {}
        a_val, b_val = 1 / (1 - delta) , 1 - delta

        # NOTE: 2 color case 
        for var, bucket_dict in attributes.items():
            alpha[var] = a_val*representation[var] 
            beta[var] = b_val*representation[var] 



        fp_color_flag = prob_vals
        fp_alpha = alpha
        fp_beta = beta



        # Solves partial assignment and then performs rounding to get integral assignment
        t1 = time.monotonic()
        res = fair_partial_assignment_2_color(df, cluster_centers, fp_alpha, fp_beta, fp_color_flag, clustering_method, num_colors, L)
        t2 = time.monotonic()
        lp_time = t2 - t1



        ### Output / Writing data to a file
        # output is a dictionary which will hold the data to be written to the
        #   outfile as key-value pairs. Outfile will be written in JSON format.
        output = {}

        # num_clusters for re-running trial
        output["num_clusters"] = num_clusters

        # Whether or not the LP found a solution
        output["partial_success"] = res["partial_success"]

        # Nonzero status -> error occurred
        output["partial_status"] = res["partial_status"]
        
        #output["dataset_distribution"] = dataset_ratio

        # Save alphas and betas from trials
        output['prob_proportions'] = representation
        output["alpha"] = alpha
        output["beta"] = beta



        # Save size of each cluster
        output["sizes"] = sizes

        output["attributes"] = attributes


        # These included at end because their data is large
        # Save points, colors for re-running trial
        # Partial assignments -- list bc. ndarray not serializable
        output["centers"] = [list(center) for center in cluster_centers]
        output["points"] = [list(point) for point in df.values]

        # Save original clustering score
        output["unfair_score"] = initial_score
        # Original Color Blind Assignments 
        if type(pred) is not list:
            pred = pred.tolist() 
        
        output["unfair_assignments"] = pred 


        # Record Assignments 
        output["partial_assignment"] = res["partial_assignment"]
        output["assignment"] = res["assignment"]

        # Clustering score after addition of fairness
        output["objective"] = res["objective"]
        
        # Clustering score after initial LP
        output["partial_objective"] = res["partial_objective"]


 
        output['prob_values'] = prob_vals
 

        # Record Lower Bound L
        output['Cluster_Size_Lower_Bound'] = L

        # Record Classifier Accurecy 
        output['p_acc'] = p_acc

        # Record probability vecs
        output["name"] = dataset
        output["clustering_method"] = clustering_method
        output["scaling"] = scaling
        output["delta"] = delta
        output["time"] = lp_time
        output["cluster_time"] = cluster_time


        # Writes the data in `output` to a file in data_dir
        write_fairness_trial(output, data_dir)

        # Added because sometimes the LP for the next iteration solves so 
        # fast that `write_fairness_trial` cannot write to disk
        time.sleep(1) 

        return output  
