# Bayesian network for detecting freely-moving thoughts.
# Caleb Bessit and Matthew Dean
# 02 October 2025

from pylab import *
import pyagrum as gum
import matplotlib.pyplot as plt
import pyagrum.lib.image as gumimage
import os

# Create network

DOING_DECISIONS = False

if DOING_DECISIONS:
    bn = gum.InfluenceDiagram()
else:
    bn = gum.BayesNet('FreelyMovingThoughts')

parent_node             = ['freely_moving_thoughts']
alpha_features          = ['alpha_var','alpha_kurt','alpha_shan']
erp_features            = ['p3_kurt']
behavioural_features    = ['rt_mean','rt_var','ra']
other_phys_features     = ['pupil_dil']

env_features            = ['large_es']
three_value_features    = ['env']

# Create layers of features
feature_layer_1 = alpha_features + behavioural_features + other_phys_features + env_features
feature_layer_2 = erp_features

names = parent_node + feature_layer_1 + feature_layer_2


id_freely_moving, id_alpha_var, id_alpha_kurt, id_alpha_shan, id_rt_mean, id_rt_var, id_ra, id_pupil_dil, id_large_es ,id_p3_kurt= [bn.add(name, 2) for name in names]

id_env = bn.add('env', 3) 

# Other, completely independent nodes
id_history = bn.add('history', 3)       # The history of whether the person has had freely moving thoughts. 0 = they are not having freely moving thoughts or have had freely-moving thoughts for less than 5 seconds, 1 = they have had freely moving thoughts for the last 5 seconds, 2 = they have had freely moving thoughts for the last 10 seconds or longer
id_load    = bn.add('load_type',2)      # The load type of the task the subject may be doing: normal cognitive load or high cognitive load

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

    # FMT -> LES -> env
    (id_freely_moving, id_large_es),
    (id_large_es, id_env)
]
# Create arcs
for link in arc_links:
    bn.addArc(*link)

# Extension for decision network: if we observe freely moving thoughts, we should nudge the user to get their attention
if DOING_DECISIONS:
    id_decision = bn.addDecisionNode("notify_user", 2)
    id_utility  = bn.addUtilityNode("utility")

    # Whether the subject is having freely moving thoughts right now, how long they have had freely moving thoughts for and the type of task they are doing influences the action we take
    bn.addArc("freely_moving_thoughts","utility")
    bn.addArc("history", "utility")
    bn.addArc("load_type", "utility")
    bn.addArc("notify_user","utility")

# Do it in layers: start with parent node, then for every node in feature layer 1 and then every node in feature layer 2
# Parent node:
fmt_prob = 0.495260663507109
bn.cpt(parent_node[0]).fillWith([1-fmt_prob,fmt_prob])

# Hard code values based on assumptions
bn.cpt("load_type").fillWith([0.8, 0.2])
bn.cpt("history").fillWith([fmt_prob, 0.45,0.35])
# Extract conditional probabilities
probs = []
with open("prob.txt") as prob_file:
    for line in prob_file:
        probs.append(line.split(','))
probs = np.array(probs)

# Calculated from the dataset
ocular_probs = {'pupil_dil': [[1-0.771024986863271,0.771024986863271],[1-0.7837729834522977,0.7837729834522977]]}

# Hard code probabilities for large external stimulus, environment variables
bn.cpt('large_es')[{'freely_moving_thoughts':1}] = [0.9,0.1]    #If we are having freely moving thoughts, it is highly unlikely that there is a large external stimulus
bn.cpt('large_es')[{'freely_moving_thoughts':0}] = [0.48,0.52]  #If we are not having freely moving thoughts, it is likely, though not super likely, that there is a large external stimulus. 

bn.cpt('env')[{'large_es':1}] = [0.15,0.35,0.5]  #If there is a large external stimulus, we are most likely in a high-disruption-level environment, second-most likely in a medium-disruption-level environment, and least likely in a low-disruption-level environment
bn.cpt('env')[{'large_es':0}] = [0.5,0.35,0.15]


# # Feature layer one: has parent node as only parent
for child in alpha_features + behavioural_features + other_phys_features:
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
    # Utility values below were assigned by reasoning though the implications of each assignment in the context of notifying a user
    bn.utility("utility")[{"freely_moving_thoughts":1, "load_type":1, "history":2, "notify_user":1}] = 100    #If user is and has been distracted for an extended period of time due to FMT and is doing a high load task, they absolutely should be notified
    bn.utility("utility")[{"freely_moving_thoughts":1, "load_type":1, "history":1, "notify_user":1}] = 95     #If user is and has been distracted for a short while due to FMT and is doing a high load task, they should be notified
    bn.utility("utility")[{"freely_moving_thoughts":1, "load_type":1, "history":0, "notify_user":1}] = 90     #If user is distracted due to FMT and is doing a high load task, theyshould be notified
    bn.utility("utility")[{"freely_moving_thoughts":1, "load_type":0, "history":2, "notify_user":1}] = 95     #If user is and has been distracted for an extended period of time due to FMT and is doing a low load task, they should be notified, but arguably, since it is a normal load task, having some level of distractedness is permissible compared to a high-load task
    bn.utility("utility")[{"freely_moving_thoughts":1, "load_type":0, "history":1, "notify_user":1}] = 90     #If user is and has been distracted for a short while due to FMT and is doing a low load task, they should be notified, but arguably, since it is a normal load task, having some level of distractedness is permissible compared to a high-load task
    bn.utility("utility")[{"freely_moving_thoughts":1, "load_type":0, "history":0, "notify_user":1}] = 85     #If user is distracted due to FMT and is doing a low load task, they should be notified, but arguably, since it is a normal load task, having some level of distractedness is permissible compared to a high-load task
    
    bn.utility("utility")[{"freely_moving_thoughts":0, "load_type":1, "history":2, "notify_user":1}] = 60     #If the user is not currently distracted but were for an extended period prior to this instant, they should probably still be notified to redirect their attention
    bn.utility("utility")[{"freely_moving_thoughts":0, "load_type":1, "history":1, "notify_user":1}] = 50     #If the user is not currently distracted but were for a short while prior to this instant, they should probably still be notified to redirect their attention
    bn.utility("utility")[{"freely_moving_thoughts":0, "load_type":1, "history":0, "notify_user":1}] = -100   #If the user is not distracted and they were not distracted for an extended period of time leading up to this instance, they absolutely should NOT be notified. We do not want to distract them.
    bn.utility("utility")[{"freely_moving_thoughts":0, "load_type":0, "history":2, "notify_user":1}] = 50     #If the user is not currently distracted but were for an extended period prior to this instant, they should probably still be notified to redirect their attention
    bn.utility("utility")[{"freely_moving_thoughts":0, "load_type":0, "history":1, "notify_user":1}] = 40     #If the user is not currently distracted but were for a short while prior to this instant, they should probably still be notified to redirect their attention
    bn.utility("utility")[{"freely_moving_thoughts":0, "load_type":0, "history":0, "notify_user":1}] = -80    #If the user is not distracted and they were not distracted for an extended period of time leading up to this instance, they absolutely should NOT be notified. We do not want to distract them.
    
    bn.utility("utility")[{"freely_moving_thoughts":1, "load_type":1, "history":2, "notify_user":0}] = -100   #If the user is distracted due to having FMT and has had FMT for an extended period prior to this instant, and we are performing a high-load task, it is absolutely bad if we do not notify them
    bn.utility("utility")[{"freely_moving_thoughts":1, "load_type":1, "history":1, "notify_user":0}] = -95    #If the user is distracted due to having FMT and has had FMT for a short while prior to this instant, and we are performing a high-load task, it is absolutely bad if we do not notify them
    bn.utility("utility")[{"freely_moving_thoughts":1, "load_type":1, "history":0, "notify_user":0}] = -90    #If the user is distracted due to having FMT at this instant, and they are performing a high-load task, it is absolutely bad if we do not notify them
    bn.utility("utility")[{"freely_moving_thoughts":1, "load_type":0, "history":2, "notify_user":0}] = -95    #The same reasoning as the three above except it is a normal load task.
    bn.utility("utility")[{"freely_moving_thoughts":1, "load_type":0, "history":1, "notify_user":0}] = -90    
    bn.utility("utility")[{"freely_moving_thoughts":1, "load_type":0, "history":0, "notify_user":0}] = -85   
    
    bn.utility("utility")[{"freely_moving_thoughts":0, "load_type":1, "history":2, "notify_user":0}] = -5    #If the user is not having freely moving thoughts but did for an extended period of time prior to this instant and we do not notify them, it is not a horrible outcome but we probably should notify them.
    bn.utility("utility")[{"freely_moving_thoughts":0, "load_type":1, "history":1, "notify_user":0}] = -2    #Same as above and for below.
    bn.utility("utility")[{"freely_moving_thoughts":0, "load_type":0, "history":2, "notify_user":0}] = -3    
    bn.utility("utility")[{"freely_moving_thoughts":0, "load_type":0, "history":1, "notify_user":0}] =  -1 
    bn.utility("utility")[{"freely_moving_thoughts":0, "load_type":1, "history":0, "notify_user":0}] = 100   #If They are performing a high-load task, are not and have not been experiencing FMT, and we do not notify them, this is absolutely good since we do not distract them.  
    bn.utility("utility")[{"freely_moving_thoughts":0, "load_type":0, "history":0, "notify_user":0}] = 95    #Same as above
    
    

    gumimage.export(bn, "bayesian_networks/network.pdf")
    bn.saveBIFXML(  os.path.join(outdir,f"FreelyMovingThoughts.bifxml") )
    # gum.saveBN(bn, os.path.join(outdir, f"FreelyMovingThoughts.bifxml"))
else:
    gum.saveBN(bn, os.path.join(outdir, f"FreelyMovingThoughts.bif"))

gumimage.export(bn, "bayesian_networks/network.pdf")
print("Network constructed.")
