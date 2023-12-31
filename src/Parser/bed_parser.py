
import sys
import os

# Add the directory containing the module to the sys.path list
module_directory = os.path.abspath('../')


sys.path.insert(0, module_directory)
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import defaultdict
import re
import pandas as pd 
import numpy as np
from src.Parser.parser_helpers import get_keys, remove_dups, reverse_complementary_sequence, sequence_len_tuner, progress_bar
from src.Parser.parser_helpers import all_chr_dict, positive_chr_dict, create_control_seq, create_control_seq_y



def parse_bed_file(**kwargs):     # Main parser function

    fasta_path =kwargs.pop('dm6_fasta_path') 
    bed_path = kwargs.pop('bed_path')
    records_dict = kwargs.pop('records_dict')
    show_sequence_legth = kwargs.pop("show_sequence_legth")
    sequence_length = kwargs.pop("sequence_length")
    chromosome_dict = kwargs.pop('chromosome_dict')
    output_path = kwargs.pop('output_path')

    assert sequence_length % 2 != 0, "Requested length of the sequence must be an odd number eg. 249 "

    sequence_len = sequence_len_tuner(sequence_length)


    remove_dups(bed_path)

    table=pd.read_csv("data/parsed_data/Flybase_dm6_TSSs_cleaned.bed",delimiter='\t')


    positive_chr_dictionary = positive_chr_dict(chromosome_dict, table, sequence_len)

    all_chr_dictionary = all_chr_dict(chromosome_dict, table, sequence_len)



    control_seq_dict={} 

    for chr_name in chromosome_dict.keys():
        if(chr_name=="chrY"):
            control_seq_dict["chrY"]=create_control_seq_y(chromosome_dict, records_dict,sequence_len )
        else:
            control_seq_dict[chr_name]=create_control_seq(chr_name,positive_chr_dictionary,chromosome_dict, records_dict, all_chr_dictionary,sequence_len)
    for key in control_seq_dict.keys():
        print(key+": "+str(len(control_seq_dict[key])))    
    with open("data/parsed_data/Flybase_dm6_TSSs_cleaned.bed","a") as f:
        counter=0
        for key in control_seq_dict.keys():
            for seq in control_seq_dict[key]:
                f.write(key+"\t"+str(seq+sequence_len)+"\t"+str(seq+sequence_len)+"\t"+str(counter)+"\t.\t+\n")
                counter+=1


    table=pd.read_csv("data/parsed_data/Flybase_dm6_TSSs_cleaned.bed",delimiter='\t')

    if show_sequence_legth:   
    
        df = pd.DataFrame(columns=[ 'TSS','seqnames','start','end','strand','id','sequence','sequence_len'])
    else: 
        df = pd.DataFrame(columns=['TSS','seqnames','start','end','strand','id','sequence'])

    entry_array=[]

    total_items = len(table)
    
    for i in range(len(table)):
        progress_bar(i + 1, total_items, prefix='Progress:', suffix='Parsing the TSS .bed File', length=30)
        strand=records_dict[chromosome_dict[table.iloc[i][0]]].seq
        start_pos=table.iloc[i][1]-sequence_len
        end_pos=table.iloc[i][1]+sequence_len
        sequence="".join(strand[start_pos:end_pos+1].upper())
        if(table.iloc[i][5]=="-"):
            sequence=reverse_complementary_sequence(sequence)
        if(table.iloc[i][3][0]=="F"):
            label="TSS"
            tss_value=1
        else:
            label="neg"
            tss_value=0
        id=str(table.iloc[i][3])+"_"+table.iloc[i][0]+":"+str(start_pos)+":"+table.iloc[i][5]+"_"+label

        if show_sequence_legth:
            entry={ "TSS":tss_value,"seqnames":table.iloc[i][0],"start":start_pos,"end":end_pos,"strand":table.iloc[i][5],"id":id,"sequence":sequence, "sequence_len": len(sequence)}
        else: 
            entry={"TSS":tss_value,"seqnames":table.iloc[i][0],"start":start_pos,"end":end_pos,"strand":table.iloc[i][5],"id":id,"sequence":sequence}
            
        entry_array.append(entry) 


    df=df._append(entry_array,ignore_index=True)
    df_shuffled = df.sample(frac=1).reset_index(drop=True)

    pars_path = output_path + "parsed_bed_data.csv"


    df_shuffled.to_csv(pars_path,index=True)

    return pars_path

    