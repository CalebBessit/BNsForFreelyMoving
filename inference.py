# Test the utility of the classification network
# Caleb Bessit
# 05 October 2025

import os
import itertools
import numpy as np
import pandas as pd
import pyagrum as gum
import pyagrum.lib.image as gumimage
import pyagrum.lib.notebook as gnb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score



#Load network and test data

DOING_DECISIONS = True

if DOING_DECISIONS:
    bn = gum.loadID( os.path.join("bayesian_networks","FreelyMovingThoughts.bifxml") )
else:
    bn = gum.loadBN( os.path.join("bayesian_networks","FreelyMovingThoughts.bif") )


subject_range = (10,15)

# Specify indices of behavioural features 

original_features = ['p3_mean', 'p3_std', 'alpha_mean', 'alpha_var',
            'rt_mean', 'rt_var', 'ra', 'p3_log',
            'p3_shan', 'p3_sure', 'p3_skew', 'p3_kurt', 
            'alpha_log', 'alpha_shan', 'alpha_sure', 'alpha_skew', 
            'alpha_kurt']

alpha_features      = ['alpha_var','alpha_kurt','alpha_shan']
behavioral_features = ['rt_mean','rt_var','ra']
erp_features        = ['p3_kurt']


# Check pupil dilation values
scenarios = [
    {"env":0, "large_es":0, "pupil_dil":0},
    {"env":2, "large_es":1, "pupil_dil":1},
    {"env":1, "large_es":0, "pupil_dil":1},
    {"env":0}, {"env":1}, {"env":2},
    {"large_es":0}, {"large_es":1}
]

scenario_names = ["library", "busy_street", "working_in_office","quiet","medium","loud","no_distractions","distraction"]

    

chosen_features = alpha_features + behavioral_features + erp_features

feat_index = dict( zip( chosen_features, [original_features.index(feature) for feature in chosen_features] ) )


overall_labels, overall_features = None, None
for subject in range(subject_range[0], subject_range[1]):
    subject_id = f"{subject+1:02d}"
    subject_data = np.load(f"test_bin_digits/S1{subject_id}_binary_digit_data.npy")

    if overall_labels is None:
        overall_labels   = subject_data[:,3]
        overall_features = subject_data[:,4:] 
    else:
        overall_labels   = np.concatenate( (overall_labels, subject_data[:,3] ))
        overall_features = np.vstack( (overall_features, subject_data[:,4:]))  

overall_labels = np.array([0 if label==3.0 else 1 for label in overall_labels])
trials = overall_labels.shape[0]


#################################
### INFERENCE
#################################

def posterior_prob_situations(target_var, evidence, situation):
    ie = gum.LazyPropagation(bn)  
    if evidence:
        ie.setEvidence(evidence)
    ie.makeInference()

    try:
        gumimage.exportInference(bn, f"bayesian_networks/bn_{situation}.pdf", evs=evidence)
    except Exception as e:
        print(f"An error occurred: {e}.")
        

    post = ie.posterior(target_var).toarray() 
    # print(post) 
    return float(post[1]) 

def posterior_prob(target_var, evidence, index):
    ie = gum.LazyPropagation(bn)  
    if evidence:
        ie.setEvidence(evidence)
    ie.makeInference()

    if index==0:
        try:
            gumimage.exportInference(bn, f"bayesian_networks/bn_{"_".join( list(evidence.keys()))}.pdf", evs=evidence)
        except Exception as e:
            print(f"An error occurred: {e}.")

    post = ie.posterior(target_var).toarray() 
    # print(post) 
    return float(post[1])  

def posterior_prob_dec(evidence):
    ie = gum.ShaferShenoyLIMIDInference(bn)

    ie.setEvidence(evidence)
    ie.makeInference()

    p_action = ie.posterior("notify_user").toarray()

    notified_user = 1 if float(p_action[1])>0.5 else 0

    gumimage.exportInference(bn, f"bayesian_networks/dn_indiv_{"_".join(list(evidence.keys()))}.pdf", evs=evidence)

    return notified_user
def evidence_from_trial(i, subset):
    ev = {}
    row = overall_features[i]
    for feat in subset:
        ev[feat] = int(row[feat_index[feat]])
    return ev

def decision_analysis(evidence_subset):
    ie = gum.ShaferShenoyLIMIDInference(bn)
    results = []

    trues = overall_labels.tolist()
    preds = []


    scenario_preds = [[], [], []]
    for i in range(trials):
        ev = evidence_from_trial(i, evidence_subset)
        ie.setEvidence(ev)
        ie.makeInference()

        p_action = ie.posterior("notify_user").toarray()

        notified_user = 1 if float(p_action[1])>0.5 else 0
        preds.append(notified_user)
        true_val = trues[i]
 

        results.append({
            "trial": i,
            "evidence": ev,
            "notified_user" : notified_user,
            "p_action0": float(p_action[0]),
            "p_action1": float(p_action[1])
        })

        if i==0:
            # gnb.getInfluenceDiagram(bn, filename="bayesian_networks/decision_network_with_evidence.pdf", engine="dot")
            gumimage.exportInference(bn, f"bayesian_networks/dn_corr_{"_".join(ev)}.pdf", evs=ev)
        # print(results, "\n")

        for scenario_idx in range(3):
            evidence = scenarios[scenario_idx]
            
            merged_ev = ev
            for key, value in evidence.items():
                merged_ev[key] = value
            # merged_ev = ev | evidence

            ie.setEvidence(merged_ev)
            ie.makeInference()


            p_action = ie.posterior("notify_user").toarray()
            if i==0:
                gumimage.exportInference(bn, f"bayesian_networks/dn_corr_{"_".join(merged_ev)}_sc_{scenario_idx}.pdf", evs=ev)
        
            notified_user = 1 if float(p_action[1])>0.5 else 0
            scenario_preds[scenario_idx].append(notified_user)

    # metrics
    # print(trues, preds)
    acc = accuracy_score(trues, preds)
    bal_acc = balanced_accuracy_score(trues, preds)
    prec = precision_score(trues, preds, zero_division=0)
    rec = recall_score(trues, preds, zero_division=0)
    f1 = f1_score(trues, preds, zero_division=0)
    


    df = pd.DataFrame(results)
    prop_notify = (df["notified_user"].sum())/len(df)

    print(f"Using {evidence_subset} as evidence, notify_user=1 is optimal action in {prop_notify*100:.2f}% of cases.")
    results = {
        'subset': evidence_subset,
        'acc': acc,
        'bal_acc': bal_acc,
        'prec': prec,
        'rec': rec,
        'f1': f1,
    }
    # print(result)

    print(f"\n==== DECSISION RESULTS with {evidence_subset} ======")

    for scenario_idx in range(3):
        preds = np.sum(np.array(scenario_preds[scenario_idx]))
        name = scenario_names[scenario_idx]

        print(f"\t + In {name}, we notify {(preds*100)/len(scenario_preds[scenario_idx]):.2f}% of the time.")

    return results

def evaluate_subset_across_trials(subset):
    preds = []
    probs = []
    trues = overall_labels.tolist()

    for i in range(trials):
        ev = evidence_from_trial(i, subset)
        p1 = posterior_prob('freely_moving_thoughts', ev, i) #Can change targets here based on evidence and see how things go
        prob = p1
        pred = 1 if prob >= 0.5 else 0
        preds.append(pred)
        probs.append(prob)

        
    # metrics
    acc = accuracy_score(trues, preds)
    bal_acc = balanced_accuracy_score(trues, preds)
    prec = precision_score(trues, preds, zero_division=0)
    rec = recall_score(trues, preds, zero_division=0)
    f1 = f1_score(trues, preds, zero_division=0)

    # avg confidence: mean probability assigned to the chosen class
    chosen_confidences = []
    for prob, pred in zip(probs, preds):
        chosen_confidences.append(prob if pred==1 else (1.0-prob))
    avg_confidence = float(np.mean(chosen_confidences))
    result = {
        'subset': subset,
        'n_trials': trials,
        'acc': acc,
        'bal_acc': bal_acc,
        'prec': prec,
        'rec': rec,
        'f1': f1,
        'avg_confidence': avg_confidence,
    }
    return result, {'preds': preds, 'probs': probs} 


if DOING_DECISIONS:
    results = []

    # Do inference with no evidence

    res_no_evidence = decision_analysis([])
    res_no_evidence['label'] = 'no_evidence'
    results.append(res_no_evidence)
    print("No evidence:", res_no_evidence)

    # # Do inference using behavioural evidence only

    beh_subsets = []
    for i in range(1, len(behavioral_features)+1):
        for combination in itertools.combinations(behavioral_features, i):
            beh_subsets.append(tuple(combination))
    print(f"Testing {len(beh_subsets)} behavioural subsets...")

    beh_results = []
    for subset in beh_subsets:
        result = decision_analysis(list(subset))
        result['label'] = 'behavioural'
        results.append(result)
        beh_results.append(result)

    # Do inference using physiological evidence only
    phys_feats = alpha_features + erp_features
    phys_subsets = []
    for i in range(1, len(phys_feats)+1):
        for combinations in itertools.combinations(phys_feats, i):
            phys_subsets.append(tuple(combinations))
    print(f"Testing {len(phys_subsets)} physiological subsets...")

    phys_results = []
    for subset in phys_subsets:
        result = decision_analysis(list(subset))
        result['label'] = 'physiological'
        results.append(result)
        phys_results.append(result)

    # Do inference using all evidence
    res_all = decision_analysis(chosen_features)
    res_all['label'] = 'all_features'
    results.append(res_all)
    print("All-features result:", res_all)

    # Save results
    outdir = "results"
    os.makedirs(outdir, exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(['acc'], ascending=[False])
    out_csv = "decision_results_summary.csv"
    df_results.to_csv( os.path.join(outdir, out_csv), index=False)
    print(f"Saved summary to {os.path.join(outdir, out_csv)}.")

    # Doing extra info check

    print("Conflicting evidence: should probably be zero, but may be one")
    posterior_prob_dec({"history":0,"load_type":1, "alpha_shan":1, "alpha_kurt":1,"p3_kurt":1})
    
    print("All agreeing evidence: should be one")
    print(posterior_prob_dec({"history":2,"load_type":1, "alpha_shan":1, "alpha_kurt":1,"p3_kurt":1}))
    # print(posterior_prob_dec({"history":1,"load_type":1}))

else:
    results = []

    # Do inference with no evidence

    res_no_evidence, details_no = evaluate_subset_across_trials([])
    res_no_evidence['label'] = 'no_evidence'
    results.append(res_no_evidence)
    print("No evidence:", res_no_evidence)

    # Do inference using behavioural evidence only

    beh_subsets = []
    for i in range(1, len(behavioral_features)+1):
        for combination in itertools.combinations(behavioral_features, i):
            beh_subsets.append(tuple(combination))
    print(f"Testing {len(beh_subsets)} behavioural subsets...")

    beh_results = []
    for subset in beh_subsets:
        result, _ = evaluate_subset_across_trials(list(subset))
        result['label'] = 'behavioural'
        results.append(result)
        beh_results.append(result)

    # Do inference using physiological evidence only
    phys_feats = alpha_features + erp_features
    phys_subsets = []
    for i in range(1, len(phys_feats)+1):
        for combinations in itertools.combinations(phys_feats, i):
            phys_subsets.append(tuple(combinations))
    print(f"Testing {len(phys_subsets)} physiological subsets...")

    phys_results = []
    for subset in phys_subsets:
        result, _ = evaluate_subset_across_trials(list(subset))
        result['label'] = 'physiological'
        results.append(result)
        phys_results.append(result)

    # Do inference using all evidence
    res_all, details_all = evaluate_subset_across_trials(chosen_features)
    res_all['label'] = 'all_features'
    results.append(res_all)
    print("All-features result:", res_all)

    # Save results
    outdir = "results"
    os.makedirs(outdir, exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(['acc'], ascending=[False])
    out_csv = "inference_results_summary.csv"
    df_results.to_csv( os.path.join(outdir, out_csv), index=False)
    print(f"Saved summary to {os.path.join(outdir, out_csv)}.")


    # Evaluating effect of elicited variables on other unobserved variables

    
    print("\n ==== SCENARIO BASED OUTCOMES =======")
    for idx in range(len(scenarios)):
        evidence, name = scenarios[idx], scenario_names[idx]

        print(f"\t $$$ In {name} $$$")

        results = {}
        for feature in ['freely_moving_thoughts'] + alpha_features + erp_features + behavioral_features :
            value = posterior_prob_situations(feature, evidence, name)
            results[feature] = value
        
        print(f"\t\t + {results}")


        

    # posterior_prob("ra", {"large_es":0, ""})

    # env and stimulus on behaviour

    # env and 

