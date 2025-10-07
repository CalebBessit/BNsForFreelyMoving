# Bayesian network for detecting freely-moving thoughts.
# Caleb Bessit
# 02 October 2025

from pylab import *
import pyagrum as gum
import matplotlib.pyplot as plt
import pyagrum.lib.image as gumimage
import os

# Create network

DOING_DECISIONS = True

if DOING_DECISIONS:
    bn = gum.InfluenceDiagram()
else:
    bn = gum.BayesNet('FreelyMovingThoughts')

parent_node             = ['freely_moving_thoughts']
alpha_features          = ['alpha_var','alpha_kurt','alpha_shan']
erp_features            = ['p3_kurt']
behavioural_features    = ['rt_mean','rt_var','ra']
other_features          = ['pupil_dil']

# Create layers of features
feature_layer_1 = alpha_features + behavioural_features + other_features
feature_layer_2 = erp_features

names = parent_node + feature_layer_1 + feature_layer_2


id_freely_moving, id_alpha_var, id_alpha_kurt, id_alpha_shan, id_rt_mean, id_rt_var, id_ra, id_pupil_dil, id_p3_kurt = [bn.add(name, 2) for name in names]


arc_links = [
    # FMT -> alphas
    (id_freely_moving, id_alpha_var),
    (id_freely_moving, id_alpha_kurt),
    (id_freely_moving, id_alpha_shan),

    # FMT -> behavioural
    (id_freely_moving, id_rt_mean),
    (id_freely_moving, id_rt_var),
    (id_freely_moving, id_ra),

    # FMT -> pupil dilation
    (id_freely_moving, id_pupil_dil),

    # FMT and Alpha features -> ERP feature
    # (id_freely_moving, id_p3_kurt),

    (id_alpha_var, id_p3_kurt),
    (id_alpha_kurt, id_p3_kurt),
    (id_alpha_shan, id_p3_kurt),
]
# Create arcs
for link in arc_links:
    bn.addArc(*link)

# Extension for decision network: if we observe freely moving thoughts, we should nudge the user to get their attention
if DOING_DECISIONS:
    id_decision = bn.addDecisionNode("notify_user", 2)
    id_utility  = bn.addUtilityNode("utility")

    bn.addArc("freely_moving_thoughts","utility")
    bn.addArc("notify_user","utility")

# Topology created, now create probability tables.

# Do it in layers: start with parent node, then for every node in feature layer 1 and then every node in feature layer 2
# Parent node:
fmt_prob = 0.495260663507109
bn.cpt(parent_node[0]).fillWith([1-fmt_prob,fmt_prob])

# Extract conditional probabilities
probs = []
with open("prob.txt") as prob_file:
    for line in prob_file:
        probs.append(line.split(','))
probs = np.array(probs)

ocular_probs = {'pupil_dil': [[1-0.771024986863271,0.771024986863271],[1-0.7837729834522977,0.7837729834522977]]}

# # Feature layer one: has parent node as only parent
for child in feature_layer_1:
    try:
        prob = probs[(probs[:, 0] == "freely_moving_thoughts") & (probs[:, 1] == child)][0]
        bn.cpt(child)[{parent_node[0]: 0}] = [float(prob[2]), 1-float(prob[2])]
        bn.cpt(child)[{parent_node[0]: 1}] = [float(prob[3]), 1-float(prob[3])]
    except:
        prob = ocular_probs[child]
        bn.cpt(child)[{parent_node[0]: 0}] = prob[0]
        bn.cpt(child)[{parent_node[0]: 1}] = prob[1]

# # Feature layer 2: P3 node which has alpha nodes as parents
for i in range (8):
    bitmap              = [int(digit) for digit in f"{bin(i)[2:].zfill(3)}"] #Take index i and convert it to a list of binary digits. This helps us cover all combinations of alpha features being on/off
    features_and_states = dict(zip(alpha_features, bitmap))  #We have a dictionary of alpha_feature:value, e.g. when i = 5, the binary representation is 101, so we have {'alpha_var':1, 'alpha_kurt':0, 'alpha_shan':1}
    final_prob = 1
    for j in range(len(bitmap)):
        prob = probs[(probs[:, 0] == alpha_features[j]) & (probs[:, 1] == erp_features[0])][0]
        if bitmap[j]:
            final_prob *= float(prob[3])
        else:
            final_prob *= float(prob[2])
    bn.cpt(erp_features[0])[features_and_states] = [final_prob,1-final_prob]

outdir = "bayesian_networks/"
os.makedirs(outdir, exist_ok=True)


# Table for utility function
if DOING_DECISIONS:
    bn.utility("utility")[{"freely_moving_thoughts":1, "notify_user":1}] = 10    #notifying the user when they have freely-moving thoughts is very good
    bn.utility("utility")[{"freely_moving_thoughts":0, "notify_user":0}] = 5     #not notifying the user when they are not having freely-moving thoughts is also good (neutral?)
    bn.utility("utility")[{"freely_moving_thoughts":1, "notify_user":0}] = -1    #not notifying the user when they are having freely-moving thoughts is not good
    bn.utility("utility")[{"freely_moving_thoughts":0, "notify_user":1}] = -10   #notifying the user when they are not having freely-moving thoughts (potentially distracting them) is very bad

    gumimage.export(bn, "bayesian_networks/network.pdf")
    bn.saveBIFXML(  os.path.join(outdir,f"FreelyMovingThoughts.bifxml") )
    # gum.saveBN(bn, os.path.join(outdir, f"FreelyMovingThoughts.bifxml"))
else:
    gum.saveBN(bn, os.path.join(outdir, f"FreelyMovingThoughts.bif"))
