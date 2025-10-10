# Plots the results from experiments
# Caleb Bessit
# 09 October 2025

import numpy as np
import matplot2tikz
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

data = pd.read_csv("results/inference_results_summary.csv")
data = data.sort_values(by='bal_acc', ascending=False)
limit = 5

positions = np.arange(limit)
acc_pos, bal_pos = positions - 0.1, positions + 0.1

labels = ["RA", "Mean RT + RA", "Var. + Shan. + P3 Kur.", 
          "Kur. + Shan. + P3 Kur.", "Kur. + Shan."]

plt.figure(figsize=(8, 4))

plt.barh(acc_pos, data.head(limit)['acc'], height=0.15, color='skyblue', edgecolor='black')
plt.barh(bal_pos, data.head(limit)['bal_acc'], height=0.15, color='orange', edgecolor='black')

# Label setup
plt.yticks(positions, labels)
plt.gca().invert_yaxis() 
plt.title("Classification performance for best feature combinations")
plt.ylabel("Combination of features")
plt.xlabel("Metric value")

acc_patch = mpatches.Patch(facecolor='skyblue', edgecolor='black', label='Accuracy')
bal_patch = mpatches.Patch(color='orange', label='Bal. accuracy')

plt.grid(alpha=0.3, axis='x')
plt.legend(handles=[acc_patch, bal_patch])
plt.tight_layout()

matplot2tikz.save("results/class_results_horiz.tex")

plt.show()
