# Calculates the probabilties from an experiment set of data
# Caleb Bessit
# 03 October 2025


import os
import numpy as np

subjects = 10

overall_fr, overall_nfr = None, None
overall_labels = None
for subject in range(subjects):
    subject_id = f"{subject+1:02d}"
    data = np.load(f"NP_Data/S1{subject_id}_data.npy")

    fr  = data[data[:,3] == 3.0][:,4:]
    nfr = data[data[:,3] == 4.0][:,4:]
    labels = data[:,3]

    if overall_fr is None:
        overall_fr, overall_nfr = fr, nfr
        overall_labels = labels
    else:
        overall_fr  = np.vstack((overall_fr, fr))
        overall_nfr = np.vstack((overall_nfr, nfr))
        overall_labels = np.concatenate( (overall_labels, labels))

fr_medians, nfr_medians = np.median(overall_fr, axis=0), np.median(overall_nfr, axis=0)
print(overall_fr.size)
# Save median data to file; will use the same thresholds for the test data
param_dir = "parameter_data"
os.makedirs(param_dir, exist_ok=True)
np.save( os.path.join(param_dir, "fr_medians.npy"), fr_medians )
np.save( os.path.join(param_dir, "nfr_medians.npy"), nfr_medians )


# Construct dictionary of features for probs
original_features = ['p3_mean', 'p3_std', 'alpha_mean', 'alpha_var',
            'rt_mean', 'rt_var', 'ra', 'p3_log',
            'p3_shan', 'p3_sure', 'p3_skew', 'p3_kurt', 
            'alpha_log', 'alpha_shan', 'alpha_sure', 'alpha_skew', 
            'alpha_kurt']

alpha_features      = ['alpha_var','alpha_kurt','alpha_shan']
behavioral_features = ['rt_mean','rt_var','ra']
erp_features        = ['p3_kurt']

chosen_features = alpha_features + behavioral_features + erp_features


all_probs = { feature:[] for feature in chosen_features}
chosen_indices = [original_features.index(feature) for feature in chosen_features]

overall_fr, overall_nfr = overall_fr[:,chosen_indices], overall_nfr[:,chosen_indices]
combined = np.vstack((overall_fr, overall_nfr))

fr_median, nfr_median = np.median(overall_fr, axis=0), np.median(overall_nfr, axis=0)
fr_diffs, nfr_diffs = np.abs(combined-fr_median), np.abs(combined-nfr_median)

bin_digits = (fr_diffs<nfr_diffs).astype(int)

overall_labels = np.reshape(overall_labels,(overall_labels.shape[0],1)) # Reshape labels so that they can be concatenated with features
combined = np.hstack((overall_labels, bin_digits)) #Recombine labels and features

fr  = combined[combined[:,0] == 3.0][:,1:]
nfr = combined[combined[:,0] == 4.0][:,1:]




# Can now calculate probabilities based on counts
all_probs["freely_moving_thoughts"] = [nfr.shape[0]/combined.shape[0], fr.shape[0]/combined.shape[0]]

feat_on_fr, feat_on_nfr   = np.sum(fr, axis=0), np.sum(nfr, axis=0)
feat_off_fr, feat_off_nfr = fr.shape[0]-feat_on_fr, nfr.shape[0]-feat_on_nfr

feat_on_fr, feat_on_nfr = feat_on_fr/fr.shape[0], feat_on_nfr/nfr.shape[0]
feat_off_fr, feat_off_nfr = feat_off_fr/fr.shape[0], feat_off_nfr/nfr.shape[0]

# Probabilities counted, now save to file

for i in range(len(chosen_indices)):
    feature = chosen_features[i]
    all_probs[feature] = [ [feat_off_nfr[i], feat_on_nfr[i]], [feat_off_fr[i],feat_on_fr[i]] ]



np.save( os.path.join(param_dir, "probabilities.npy"), np.array([all_probs]) )
print("Done.")
