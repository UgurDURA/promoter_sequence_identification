import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
df=pd.read_csv("cofactor_expression_data.csv")
np_array_all=df.to_numpy()
np_array_2R = np_array_all[np_array_all[:,2] == "chr2R"]
# print(np_array_2R)
data_range=(2,3,4,5,7)
np_array_2R_seq = np_array_2R[:,data_range]
val_2R=np_array_2R_seq[:len(np_array_2R_seq)//2]
data_range_cofs=(9,10,18,19,22)
val_2R_cofs = np_array_2R[:,data_range_cofs]
val_2R_cofs=val_2R_cofs[:len(val_2R_cofs)//2]
X_val_p65 = val_2R[val_2R_cofs[:,0]>=1]
X_val_p300 = val_2R[val_2R_cofs[:,1]>=1]     
X_val_gfzf = val_2R[val_2R_cofs[:,2]>=1]
X_val_chro = val_2R[val_2R_cofs[:,3]>=1]
X_val_mof = val_2R[val_2R_cofs[:,4]>=1]
print((X_val_p65.shape))
path_p65="./coordinates_p65.txt"
with open(path_p65, 'w') as f:
    f.write("chr,start,end,strand,seq")
    f.write('\n')
    for i in range(0,len(X_val_p65)):
        f.write(str(X_val_p65[i][0])+","+str(X_val_p65[i][1])+","+str(X_val_p65[i][2])+","+str(X_val_p65[i][3])+","+str(X_val_p65[i][4]))
        f.write('\n')
path_p300="./coordinates_p300.txt"
with open(path_p300, 'w') as f:
    f.write("chr,start,end,strand,seq")
    f.write('\n')
    for i in range(0,len(X_val_p300)):
        f.write(str(X_val_p300[i][0])+","+str(X_val_p300[i][1])+","+str(X_val_p300[i][2])+","+str(X_val_p300[i][3])+","+str(X_val_p300[i][4]))
        f.write('\n')
path_gfzf="./coordinates_gfzf.txt"
with open(path_gfzf, 'w') as f:
    f.write("chr,start,end,strand,seq")
    f.write('\n')
    for i in range(0,len(X_val_gfzf)):
        f.write(str(X_val_gfzf[i][0])+","+str(X_val_gfzf[i][1])+","+str(X_val_gfzf[i][2])+","+str(X_val_gfzf[i][3])+","+str(X_val_gfzf[i][4]))
        f.write('\n')
path_chro="./coordinates_chro.txt"
with open(path_chro, 'w') as f:
    f.write("chr,start,end,strand,seq")
    f.write('\n')
    for i in range(0,len(X_val_chro)):
        f.write(str(X_val_chro[i][0])+","+str(X_val_chro[i][1])+","+str(X_val_chro[i][2])+","+str(X_val_chro[i][3])+","+str(X_val_chro[i][4]))
        f.write('\n')
path_mof="./coordinates_mof.txt"
with open(path_mof, 'w') as f:
    f.write("chr,start,end,strand,seq")
    f.write('\n')
    for i in range(0,len(X_val_mof)):
        f.write(str(X_val_mof[i][0])+","+str(X_val_mof[i][1])+","+str(X_val_mof[i][2])+","+str(X_val_mof[i][3])+","+str(X_val_mof[i][4]))
        f.write('\n')



