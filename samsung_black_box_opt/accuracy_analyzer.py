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

#0820, 0819, 0812

df3rd = pd.read_csv("./real_submission_0820_1.csv")
df2nd = pd.read_csv("./real_submission_0819_1.csv")
df1st = pd.read_csv("./data/backup/0812/real_submission_0812.csv")

df3rd = df3rd.sort_values(by=['y'], ascending=False)
df2nd = df2nd.sort_values(by=['y'], ascending=False)
df1st = df1st.sort_values(by=['y'], ascending=False)

v3rd = df3rd['ID'].values[:500]
v2nd = df2nd['ID'].values[:500]
v1st = df1st['ID'].values[:500]

tmp_correct_IDs = []

for t3rd in v3rd:
    for t2nd in v2nd:
        if t3rd == t2nd:
            tmp_correct_IDs.append(t3rd)

final_correct_IDs = []

for t2nd in tmp_correct_IDs:
    for t1st in v1st:
        if t2nd == t1st:
            final_correct_IDs.append(t2nd)

print(final_correct_IDs)
print(len(final_correct_IDs))

cmp_path = "./real_submission.csv"
df_cmp = pd.read_csv(cmp_path)
df_cmp = df_cmp.sort_values(by=['y'], ascending=False)
v_cmp = df_cmp['ID'].values[:500]

cmp_correct_IDs = []

for t_cmp in v_cmp:
    for t_final in final_correct_IDs:
        if t_cmp == t_final:
            cmp_correct_IDs.append(t_cmp)


print(cmp_correct_IDs)
print(len(cmp_correct_IDs))