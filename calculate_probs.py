# Calculates the probabilties from an experiment set of data
# Caleb Bessit and Matthew Dean
# 03 October 2025

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import os
import math
import copy

def round_sig(x, sig=3):
    if x == 0:
        return 0
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)

info_file = open("info_prob.txt","w")   # Stores probablilities in a format that is easy for humans to read.
prob_file = open("prob.txt","w")        # Stores probabilities in a format easier for the network to read.

features = ['p3_mean', 'p3_STD', 'alpha_mean', 'alpha_var',
            'rt_mean', 'rt_var', 'ra', 'p3_log',
            'p3_shan', 'p3_sure', 'p3_skew', 'p3_kurt', 
            'alpha_log', 'alpha_shan', 'alpha_sure', 'alpha_skew', 
            'alpha_kurt']

# Can change the below up to 15
subjects = 10

# Aggregate data structure
overall_comb, overall_fr, overall_nfr = None, None, None

for i in range(subjects):
    subject_id = f"{i+1:02d}"
    print(f"Doing {subject_id}")
    
    # Load this subject's test file
    test = np.load(f"NP_Data/S1{subject_id}_data.npy")

    # Filter out the samples into those where they have freely moving thoughts (the value in column with index 3 is equal to 3.0) and 
    # those where they are NOT having freely moving thoughts (value in column with index 3 equal to 4.0)
    # Also only take the columns from index 4 onwards, because that corresponds to the feature columns. See the list of columns at the top of the file.
    comb = test[:,4:]
    fr  = test[test[:,3] == 3.0][:,4:]
    nfr = test[test[:,3] == 4.0][:,4:]

    # Concatenate to aggregate data structure
    if overall_fr is None:
        overall_comb, overall_fr, overall_nfr = comb, fr, nfr
    else:
        overall_comb  = np.vstack((overall_comb, comb))
        overall_fr  = np.vstack((overall_fr, fr))
        overall_nfr = np.vstack((overall_nfr, nfr))

info_file.write(f"P(FMT) = {len(overall_fr)}/{len(overall_comb)} = {len(overall_fr)/len(overall_comb)}\n")
info_file.write(f"P(not FMT) = {len(overall_nfr)}/{len(overall_comb)} = {len(overall_nfr)/len(overall_comb)}\n")
info_file.write("\n")

# prob_file.write(f"freely_moving_thoughts={len(overall_fr)/len(overall_comb)}\n")

#Invert matrices so that each row corresponds to a feature
overall_comb = np.transpose(overall_comb)
overall_fr = np.transpose(overall_fr)
overall_nfr = np.transpose(overall_nfr)

with open("network.txt") as network:
    #First line of file is a list of nodes connected the "freely_moving_thoughts" node
    nodes = network.readline().strip()
    
    medians=[]
    #Find the median value for each feature
    for i in range(17):
        feature = features[i]
        info_file.write(feature)
        
        i_comb = copy.deepcopy(overall_comb[i])
        i_fr = overall_fr[i]
        i_nfr = overall_nfr[i]
        
        i_comb.sort()
        total = len(i_comb)
        
        median = i_comb[int(len(i_comb)/2)]
        # info_file.write(f"{i},{median},{round_sig(median)}")
        median = round_sig(median)
        medians.append(median)
        
        lt_counts = [0,0,0] # counts num values less than median: in general, given free moving thoughs, give not free moving thoughts
        for val in i_fr:
            if val<median:
                lt_counts[0]+=1
                lt_counts[1]+=1
                
        for val in i_nfr:
            if val<median:
                lt_counts[0]+=1
                lt_counts[2]+=1
        
        info_file.write(f"P(x<{median}) = {lt_counts[0]}/{total} = {lt_counts[0]/total}\n")
        info_file.write(f"P(x<{median}|FMT) = {lt_counts[1]}/{len(i_fr)} = {lt_counts[1]/len(i_fr)}\n")
        info_file.write(f"P(x<{median}|NOT FMT) = {lt_counts[2]}/{len(i_nfr)} = {lt_counts[2]/len(i_nfr)}\n")
        info_file.write("\n")
        
        if feature in nodes:
            # prob_file.write(f"{feature}={lt_counts[0]/total}\n")
            prob_file.write(f"freely_moving_thoughts,{feature},{round_sig(lt_counts[1]/len(i_fr),15)},{round_sig(lt_counts[2]/len(i_nfr),15)}\n")            

    # Subsequent lines correspond to other arcs in the network
    info_file.write("NETWORK PROBABILITIES\n")    
    for line in network:
        src, dests = line.strip().split("->")
        
        src_index = features.index(src)
        src_list = overall_comb[src_index]
        src_median = medians[src_index]
        src_total = len(src_list)
        src_lt_count = len(src_list[src_list<src_median])
        
        for dest in dests.split(","):
            dest_index = features.index(dest)
            dest_list = overall_comb[dest_index]
            dest_median = medians[dest_index]
            
            dest_lt_counts = [0,0]  # counts num values less than median: given free moving thoughs, given not free moving thoughts
            for i in range(src_total):
                if src_list[i]<src_median:
                    if dest_list[i]<dest_median:
                        dest_lt_counts[0]+=1       
                else:
                    if dest_list[i]<dest_median:
                        dest_lt_counts[1]+=1 
            
            info_file.write(f"{src}->{dest}")
            info_file.write(f"P({dest}<{dest_median}|{src}<{src_median}) = {dest_lt_counts[0]}/{src_lt_count} = {dest_lt_counts[0]/src_lt_count}\n")
            info_file.write(f"P({dest}<{dest_median}|{src}>={src_median}) = {dest_lt_counts[1]}/{src_total-src_lt_count} = {dest_lt_counts[1]/(src_total-src_lt_count)}")
            info_file.write("\n")

            prob_file.write(f"{src},{dest},{dest_lt_counts[0]/src_lt_count},{dest_lt_counts[1]/(src_total-src_lt_count)}\n")

# Calculate probabilities of ocular data based
numSamples = [[113,88],[121,88]] # rows=fr and nfr, cols=participant 1 and 2
all_means = {'size':[[1801,1670],[2209,2025]], 'duration':[[635,329],[453,265]], 'rate':[[6.07,0.27],[4.11,0.04]]}
all_ranges = {'size':[[389,281],[378,330]], 'duration':[[392,157],[204,38]], 'rate':[[5.7,0.9],[3.92,0.38]]}

eye = open('ocularProbs.txt','w')
for key, means in all_means.items():
    ranges = all_ranges[key]
    
    mean = [0,0]
    ran = [0,0]
    for i in range(2):
        for j in range(2):
            mean[i] += means[i][j]*numSamples[i][j]
            ran[i] += ranges[i][j]*numSamples[i][j]
        mean[i] = mean[i]/sum(numSamples[i])
        ran[i] = ran[i]/sum(numSamples[i])
    midpoint = sum(mean)/2
    prob_false = (ran[0] - abs(mean[0]-midpoint))/(2*ran[0])
    eye.write(f"P({key}|fr) = {1-prob_false}\n")
    
    prob_false = (ran[1] - abs(mean[1]-midpoint))/(2*ran[1])
    eye.write(f"P({key}|nfr) = {1-prob_false}\n")
