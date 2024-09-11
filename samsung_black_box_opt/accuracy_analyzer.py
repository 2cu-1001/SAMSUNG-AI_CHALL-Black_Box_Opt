import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from copy import deepcopy

#0822 0820, 0819, 0812

sota1 = pd.read_csv("./data/backup/0909/real_submission_0909_744_1.csv")
sota2 = pd.read_csv("./data/backup/0909/real_submission_0909_744_2.csv")

sota1 = sota1.sort_values(by=['y'], ascending=False)
sota2 = sota2.sort_values(by=['y'], ascending=False)

sota1 = sota1['ID'].values[:500]
sota2 = sota2['ID'].values[:500]

tmp0_correct_IDs = []

for t1 in sota1:
    for t2 in sota2:
        if t1 == t2:
            tmp0_correct_IDs.append(t1)

tmp0_correct_IDs = sorted(tmp0_correct_IDs)
print(tmp0_correct_IDs)
print("co-sota sz :", len(tmp0_correct_IDs))

print("##################################################################\
    ##################################################################\
    ##################################################################")

cmp_path = "./real_submission.csv"
df_cmp = pd.read_csv(cmp_path)
df_cmp = df_cmp.sort_values(by=['y'], ascending=False)
v_cmp = df_cmp['ID'].values[:500]

cmp_correct_IDs = []

for t_cmp in v_cmp:
    for t_sota in tmp0_correct_IDs:
        if t_cmp == t_sota:
            cmp_correct_IDs.append(t_cmp)

cmp_correct_IDs = sorted(cmp_correct_IDs)
print(cmp_correct_IDs)
print("cur model sz", len(cmp_correct_IDs))

