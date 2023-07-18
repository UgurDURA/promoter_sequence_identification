import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def parse_motifs(path_csv,path_txt):
    df = pd.read_csv(path_csv)
    sequence_id_location_dict={}
    for i in range(0,len(df)):
        seq_id=df.iloc[i,0]
        words=seq_id.split("_")
        start=int(words[1])-1
        end=int(words[2])
        seq_tuple=(start,end)
        motif_start=int(df.iloc[i,1])-1
        motif_end=int(df.iloc[i,2])-1
        motif_tuple=(motif_start,motif_end)
        if(seq_tuple not in sequence_id_location_dict.keys()):
            sequence_id_location_dict[seq_tuple]=[motif_tuple]
        else:
            sequence_id_location_dict[seq_tuple].append(motif_tuple)
    #print(sequence_id_location_dict)
    sequence_location_dict={}
    txt_file=open(path_txt,'r')
    seq_lines=txt_file.readlines()
    seq_lines.pop(0)
    for line in seq_lines:
        splitted_line=line.split(",")
        start=int(splitted_line[1])
        end=int(splitted_line[2])
        seq_id=(start,end)
        if(seq_id in sequence_id_location_dict.keys()):
            splitted_line[4]=splitted_line[4].strip("\n")
            sequence=splitted_line[4]
            sequence_location_dict[sequence]=[]
            for val in sequence_id_location_dict[seq_id]:
                sequence_location_dict[sequence].append(val)
    return sequence_location_dict



