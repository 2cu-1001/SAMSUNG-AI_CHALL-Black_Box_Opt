import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import os


sota = pd.read_csv("./data/backup/0812/real_submission_0812.csv")
cur = pd.read_csv("./real_submission.csv")
sota = sota[['y']][:int(len(sota))]
cur = cur[['y']][:int(len(cur))]
plt.plot(sota, label='sota')
plt.plot(cur, label='cur')
result = sota.sub(cur)
plt.plot(result, label='result')

plt.legend()

plt.show()

