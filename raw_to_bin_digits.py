# Converts raw data to evidence logits which can be used by the network for inference
# 06 October 2025
# Caleb Bessit

import os
import numpy as np
# import argparse

# parser = argparse.ArgumentParser(description="Example command-line program")
# parser.add_argument("stat_index", type=int, help="Path to the input file")        # positional
# args = parser.parse_args()


subjects = (10,15)

original_features = ['p3_mean', 'p3_std', 'alpha_mean', 'alpha_var',
            'rt_mean', 'rt_var', 'ra', 'p3_log',
            'p3_shan', 'p3_sure', 'p3_skew', 'p3_kurt', 
            'alpha_log', 'alpha_shan', 'alpha_sure', 'alpha_skew', 
            'alpha_kurt']

alpha_features      = ['alpha_var','alpha_kurt','alpha_shan']
behavioral_features = ['rt_mean','rt_var','ra']
erp_features        = ['p3_kurt']

chosen_features = alpha_features + behavioral_features + erp_features
chosen_indices = [original_features.index(feature) for feature in chosen_features]

all_probs = { feature:[] for feature in chosen_features}
prob_ave = []

outdir = "test_bin_digits"
os.makedirs(outdir, exist_ok=True)
for subject in range(subjects[0], subjects[1]):
    subject_id = f"{subject+1:02d}"
    data = np.load(f"NP_Data/S1{subject_id}_data.npy")

    fr  = data[data[:,3] == 3.0][:,4:]
    nfr = data[data[:,3] == 4.0][:,4:]

    mask = data[data[:,3] == 3.0][:,3]
    
    features = data[:,4:]

    # fr_medians, nfr_medians = np.median(fr, axis=0), np.median(nfr, axis=0)
    # fr_diffs, nfr_diffs = np.abs(features-fr_medians), np.abs(features-nfr_medians)
    
    # bin_digits = (fr_diffs < nfr_diffs).astype(int)
    fr_median, nfr_median = np.median(fr, axis=0), np.median(nfr, axis=0)
    fr_mean, nfr_mean = np.mean(fr, axis=0), np.mean(nfr, axis=0)
    fr_std, nfr_std = np.std(fr, axis=0), np.std(nfr, axis=0)

    # fr_diffs, nfr_diffs = np.abs(combined-fr_median), np.abs(combined-nfr_median)
    stats = np.load(f"parameter_data/statistics.npy")
    print(stats)
    num_stats = len(stats[0])
    fr_stats, nfr_stats = [], []
    for i in range(num_stats):
        fr_stats.append([])
        nfr_stats.append([])
    
    for i in range(len(features[0])): 
        for j in range(num_stats):
            if i in chosen_indices:
                index = chosen_indices.index(i)
                fr_stats[j].append(stats[0][j][index])
                nfr_stats[j].append(stats[1][j][index])  
            else:
                fr_stats[j].append(np.inf)
                nfr_stats[j].append(np.inf)
    
    fr_min_diffs, nfr_min_diffs = None, None
    for i in range(len(fr_stats)):
        fr_diffs, nfr_diffs = np.array([fr_stats[i]])-features, np.array([nfr_stats[i]])-features
        if i==0:
            fr_diffs, nfr_diffs = -fr_diffs, -nfr_diffs   
        elif i>1: 
            fr_diffs, nfr_diffs = np.abs(fr_diffs), np.abs(nfr_diffs)
                
        fr_diffs[fr_diffs < 0], nfr_diffs[nfr_diffs < 0] = np.inf, np.inf

        if fr_min_diffs is None:
            fr_min_diffs, nfr_min_diffs = fr_diffs, nfr_diffs
        else:
            fr_min_diffs, nfr_min_diffs = np.minimum(fr_min_diffs, fr_diffs), np.minimum(nfr_min_diffs, nfr_diffs)
                
    bin_digits = (fr_min_diffs<=nfr_min_diffs).astype(int)
        
    combined_data = np.hstack( (data[:,:4], bin_digits))

    np.save( os.path.join(outdir, f"S1{subject_id}_binary_digit_data.npy"), combined_data )
    print(f"Done with {subject_id}.")
    
#     labels = data[:,3]
#     labels = np.reshape(labels,(labels.shape[0],1))
#     combined = np.hstack( (labels, bin_digits))
#     fr  = combined[combined[:,0] == 3.0][:,1:]
#     nfr = combined[combined[:,0] == 4.0][:,1:]
    
#     feat_on_fr, feat_on_nfr   = np.sum(fr, axis=0), np.sum(nfr, axis=0)
#     feat_off_fr, feat_off_nfr = fr.shape[0]-feat_on_fr, nfr.shape[0]-feat_on_nfr

#     feat_on_fr, feat_on_nfr = feat_on_fr/fr.shape[0], feat_on_nfr/nfr.shape[0]
#     feat_off_fr, feat_off_nfr = feat_off_fr/fr.shape[0], feat_off_nfr/nfr.shape[0]

#     # Probabilities counted, now save to file

#     for i in range(len(chosen_indices)):
#         feature = chosen_features[i]
#         all_probs[feature] = [ [feat_off_nfr[i], feat_on_nfr[i]], [feat_off_fr[i],feat_on_fr[i]] ]

#     prob_ave.append(np.average([val[1][1] for val in all_probs.values()]))

# valid_file = open('validation.txt','a')
# valid_file.write(f"{args.stat_index} : {np.average(prob_ave)}")
# valid_file.write("\n")
    
    # print(combined_data.shape)


    # Code below is to manually check correctness
    # if subject==10:
    #     print(f"Data:\n\t= {data[:1,:]}\n")
    #     print(f"FR, NFR medians: \n\t+ {fr_medians}, \n\t- {nfr_medians}\n")
    #     print(f"Features: \n\t> {features[:1,:]}\n")
    #     print(f"Diffs: \n\t+ {fr_diffs[:1,:]}, \n\t- {nfr_diffs[:1,:]}\n")
    #     print(f"Logits: \n\t~ {logits[:1,:]}\n")
    #     print(f"Combined: \n\t= {combined_data[:1,:]}\n")



    
