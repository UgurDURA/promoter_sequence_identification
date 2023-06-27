import pyranges as pr
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import defaultdict
import gzip
import pandas as pd
import random
import json

from pandasql import sqldf
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from collections import deque
import math
import re


from src.Parser.parser_helpers import cofactor_name_exractor,get_keys, reverse_complementary_sequence

def cofactor_expression_parser(**kwargs):

    fasta_path =kwargs.pop('fasta_path') 
    expression_path = kwargs.pop('expression_path')
    records = kwargs.pop('records')
    show_sequence_legth = kwargs.pop("show_sequence_legth")

    cofactor_dataframe = pd.read_excel(expression_path)

    cofactors = cofactor_name_exractor(cofactor_dataframe)



    id = []
    chromosome_keys = []
    start_index = []
    end_index = []
    TSS_start_index = []
    strand = []
    expression_list_oflist = []

    pattern = r"^(chr\w+?)(?:Het)?_(\d+)_(\d+)_(\d+)_(\+|-)_\w+$"

    for i in range(len(cofactor_dataframe['full_name'])):
        sentence = cofactor_dataframe['full_name'][i]
        match = re.search(pattern, sentence)

        if match:
            id.append(sentence)
            chromosome_keys.append(match.group(1))
            start_index.append(int(match.group(2)))
            end_index.append(int(match.group(3)))
            TSS_start_index .append(int(match.group(4)))
            strand.append(match.group(5))
            expressions = []
            for cofactor in cofactors:
                expression_level = cofactor_dataframe[cofactor][i]
                expressions.append(expression_level)
        
            expression_list_oflist.append(expressions)


    descriptions = get_keys(fasta_path)


    
    # TODO Need to be discussed with Monika 
    descriptions['chrM'] = 'NC_024511.2'
    descriptions['chrU']= 'NW_007931084.1'
    descriptions['chrUextra']= 'NW_007931084.1'


    list_tuple_TSS = []


    for i in range(len(id)):
        id_= id[i]
        chromosome_ = chromosome_keys[i]
        start_index_ = start_index[i]
        end_index_ = end_index[i]
        strand_ = strand[i]
        
        if chromosome_ != 'chrU' or 'chrUextra':
            gene_ID = descriptions[chromosome_]
            sequence = str(records[gene_ID].seq)

            sequence = sequence[start_index_:end_index_+1]
            sequence = sequence.upper()

            if strand == "-":
                sequence = reverse_complementary_sequence(sequence)

            
            
            
            if show_sequence_legth:
                new_row  = ("1",
                            chromosome_, 
                            str(start_index_), 
                            str(end_index_), 
                            strand_, 
                            id_, 
                            sequence, 
                            str(len(sequence)))
            else:
                new_row  = ("1",
                            chromosome_, 
                            str(start_index_), 
                            str(end_index_), 
                            strand_, 
                            id_, 
                            sequence)
            

            list_tuple_TSS.append(new_row)
    

    if show_sequence_legth:   
    
        expression_dataframe= pd.DataFrame(list_tuple_TSS, columns=['TSS','seqnames', 'start', 'end', 'strand', 'ID', 'sequence', 'sequence_len'])
    else: 
            expression_dataframe= pd.DataFrame(list_tuple_TSS, columns=['TSS','seqnames', 'start', 'end', 'strand', 'ID', 'sequence'])

    
    expression_dataframe[cofactors] = expression_list_oflist

    expression_dataframe.to_csv('data/parsed_data/cofactor_expression_data_csv',index=True)








    



