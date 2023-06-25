import random as rnd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import defaultdict
import re
import pandas as pd 
import numpy as np


path_bed = '/Users/ugur_dura/Desktop/IN2393-Machine Learning for Regulatory Genomics/Project/promoter_sequence_identification/Trial_Scripts/Flybase_dm6_TSSs.bed'
path_fasta = '/Users/ugur_dura/Desktop/IN2393-Machine Learning for Regulatory Genomics/Project/promoter_sequence_identification/Trial_Scripts/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fa'

def remove_dups(bedfile):
    unique_lines = set()
    with open(bedfile, "r") as f:
        for line in f:
            cleaned_line = line.strip()
            # print(cleaned_line)
            unique_lines.add(cleaned_line)
    output_file="Flybase_dm6_TSSs_cleaned.bed"
    with open(output_file, "w") as f:
        for line in unique_lines:
            f.write(line + "\n")
        
def complementary(strand):
    complementary= strand.replace("A", "t").replace(
            "C", "g").replace("T", "a").replace("G", "c")
    complementary = complementary.upper()
    reverse_complementary=complementary[::-1]
    return reverse_complementary


records = SeqIO.to_dict(SeqIO.parse(open(path_fasta), 'fasta'))
values=[(key,value) for key, value in records.items() if 'NC' in key or 'NT' in key]
records_dict=dict(values)
chromosome_dict={}
for (key,val) in records_dict.items():
    if("chromosome" in val.description):
        chromosome_name="chr"+re.split("chromosome ",val.description)[1]
        chromosome_dict[chromosome_name]=key
remove_dups(path_bed)
table=pd.read_csv("Flybase_dm6_TSSs_cleaned.bed",delimiter='\t')
def all_chr_dict():
    all_chr_dict={}
    for key in chromosome_dict.keys():
        all_chr_dict[key]=[]
    for i in range(0,len(table)):
            all_chr_dict[table.iloc[i][0]].append(int(table.iloc[i][1])-124)
    return all_chr_dict
def positive_chr_dict():
    positive_chr_dict={}
    for key in chromosome_dict.keys():
        positive_chr_dict[key]=[]
    for i in range(0,len(table)):
        if(table.iloc[i][5]=="+"):
             positive_chr_dict[table.iloc[i][0]].append(int(table.iloc[i][1])-124)
    return positive_chr_dict
def create_control_seq(chr_name,positive_chr_dict,all_chr_dict):
    samples=[]
    num_samples=len(all_chr_dict[chr_name])*2
    for i in range(0,num_samples):
        while(True):
            control_start=rnd.randint(0,len(records_dict[chromosome_dict[chr_name]].seq)-248)
            control_stop=control_start+248 #might be 249
            if(control_start in samples):
                continue
            for j in range(0,len(positive_chr_dict[chr_name])):
                if(((control_start>positive_chr_dict[chr_name][j]) and (control_start<positive_chr_dict[chr_name][j]+248))or 
                       ((control_stop>positive_chr_dict[chr_name][j]) and (control_stop<positive_chr_dict[chr_name][j]+248))):
                    break
            else:
                samples.append(control_start)
                break
    return samples
 
def create_control_seq_y():
    samples=[]
    num_samples=3000
    for i in range(0,num_samples):
        while(True):
            control_start=rnd.randint(0,len(records_dict[chromosome_dict["chrY"]].seq)-248)
            if(control_start in samples):
                continue
            samples.append(control_start)
            break
    return samples

positive_chr_dict=positive_chr_dict()

all_chr_dict=all_chr_dict()
control_seq_dict={} 
for chr_name in chromosome_dict.keys():
    if(chr_name=="chrY"):
        control_seq_dict["chrY"]=create_control_seq_y()
    else:
        control_seq_dict[chr_name]=create_control_seq(chr_name,positive_chr_dict,all_chr_dict)
for key in control_seq_dict.keys():
    print(key+": "+str(len(control_seq_dict[key])))    
with open("Flybase_dm6_TSSs_cleaned.bed","a") as f:
    counter=0
    for key in control_seq_dict.keys():
        for seq in control_seq_dict[key]:
            f.write(key+"\t"+str(seq+124)+"\t"+str(seq+124)+"\t"+str(counter)+"\t.\t+\n")
            counter+=1



table=pd.read_csv("Flybase_dm6_TSSs_cleaned.bed",delimiter='\t')       
df = pd.DataFrame(columns=['set','TSS','seqnames','start','end','strand','id','sequence'])
entry_array=[]
for i in range(len(table)):
    strand=records_dict[chromosome_dict[table.iloc[i][0]]].seq
    start_pos=table.iloc[i][1]-124
    end_pos=table.iloc[i][1]+124
    sequence="".join(strand[start_pos:end_pos+1].upper())
    if(table.iloc[i][5]=="-"):
        sequence=complementary(sequence)
    if(table.iloc[i][3][0]=="F"):
        label="TSS"
        tss_value=1
    else:
        label="neg"
        tss_value=0
    id=str(table.iloc[i][3])+"_"+table.iloc[i][0]+":"+str(start_pos)+":"+table.iloc[i][5]+"_"+label
    entry={'set':"train","TSS":tss_value,"seqnames":table.iloc[i][0],"start":start_pos,"end":end_pos,"strand":table.iloc[i][5],"id":id,"sequence":sequence}
    entry_array.append(entry) 
#display(df)
df=df._append(entry_array,ignore_index=True)
df.to_csv("train_data.csv",index=False)

    





