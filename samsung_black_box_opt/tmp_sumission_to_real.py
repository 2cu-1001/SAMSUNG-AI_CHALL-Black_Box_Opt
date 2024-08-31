import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas as pd
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import os
from copy import deepcopy

tmp_submission = pd.read_csv("./tmp_submission.csv")
pred = tmp_submission["y"]

submission = pd.read_csv("./data/sample_submission.csv")
submission["y"] = pred

submission.to_csv("real_submission.csv", index=False)