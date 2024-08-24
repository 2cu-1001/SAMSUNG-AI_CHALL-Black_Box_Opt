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

df5th = pd.read_csv("./data/backup/0823/real_submission_0823.csv")
df4th = pd.read_csv("./data/backup/0822/real_submission_0822.csv")
df3rd = pd.read_csv("./data/backup/0820/real_submission_0820.csv")
df2nd = pd.read_csv("./data/backup/0819/real_submission_0819.csv")
df1st = pd.read_csv("./data/backup/0812/real_submission_0812.csv")

df5th = df5th.sort_values(by=['y'], ascending=False)
df4th = df4th.sort_values(by=['y'], ascending=False)
df3rd = df3rd.sort_values(by=['y'], ascending=False)
df2nd = df2nd.sort_values(by=['y'], ascending=False)
df1st = df1st.sort_values(by=['y'], ascending=False)

v5th = df5th['ID'].values[:500]
v4th = df4th['ID'].values[:500]
v3rd = df3rd['ID'].values[:500]
v2nd = df2nd['ID'].values[:500]
v1st = df1st['ID'].values[:500]

tmp0_correct_IDs = []

for t4th in v4th:
    for t5th in v5th:
        if t5th == t4th:
            tmp0_correct_IDs.append(t5th)
print("tmp1 sz :", len(tmp0_correct_IDs))

tmp1_correct_IDs = []

for t4th in tmp0_correct_IDs:
    for t3rd in v3rd:
        if t3rd == t4th:
            tmp1_correct_IDs.append(t3rd)
print("tmp1 sz :", len(tmp1_correct_IDs))

tmp2_correct_IDs = []

for t3rd in tmp1_correct_IDs:
    for t2nd in v2nd:
        if t3rd == t2nd:
            tmp2_correct_IDs.append(t3rd)
print("tmp2 sz :", len(tmp2_correct_IDs))

final_correct_IDs = []

for t2nd in tmp2_correct_IDs:
    for t1st in v1st:
        if t2nd == t1st:
            final_correct_IDs.append(t2nd)
print("final sz :", len(final_correct_IDs))

print(final_correct_IDs)

cmp_path = "./real_submission.csv"
df_cmp = pd.read_csv(cmp_path)
df_cmp = df_cmp.sort_values(by=['y'], ascending=False)
v_cmp = df_cmp['ID'].values[:500]

cmp_correct_IDs = []

for t_cmp in v_cmp:
    for t_final in final_correct_IDs:
        if t_cmp == t_final:
            cmp_correct_IDs.append(t_cmp)

cmp_correct_IDs = sorted(cmp_correct_IDs)
print(cmp_correct_IDs)
print(len(cmp_correct_IDs))


t = []
for t_cmp in v_cmp:
    for t_final in v1st:
        if t_cmp == t_final:
            t.append(t_cmp)

t = sorted(t)
print(t)
print(len(t))