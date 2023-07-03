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



from src.Parser.parser_helpers import cofactor_name_exractor,get_keys, reverse_complementary_sequence, progress_bar, list_to_queue

def cofactor_expression_parser(**kwargs):

    fasta_path =kwargs.pop('fasta_path') 
    expression_path = kwargs.pop('expression_path')
    records_dict = kwargs.pop('records_dict')
    show_sequence_legth = kwargs.pop("show_sequence_legth")
    descriptions = kwargs.pop('chromosome_dict')

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
            if match.group(1) != 'chrU' and match.group(1) != 'chrUextra' and match.group(1) !=  'chrM':
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
    # descriptions['chrM'] = 'NC_024511.2'
    # descriptions['chrU']= 'NW_007931084.1'
    # descriptions['chrUextra']= 'NW_007931084.1'


    id = list_to_queue(id)
    chromosome_keys = list_to_queue(chromosome_keys)
    start_index = list_to_queue(start_index)
    end_index = list_to_queue(end_index)
    TSS_start_index = list_to_queue(TSS_start_index)
    strand = list_to_queue(strand)



    list_tuple_TSS = []

    total_items = id.qsize()

    for i in range(total_items):

        progress_bar(i + 1, total_items, prefix='Progress:', suffix='Parsing the Cofactor Data', length=30)
        id_= id.get()
        chromosome_ = chromosome_keys.get()
        start_index_ = start_index.get()
        end_index_ = end_index.get()
        strand_ = strand.get()
        
        if chromosome_ != 'chrU' and chromosome_ != 'chrUextra' and chromosome_ !=  'chrM':
            gene_ID = descriptions[chromosome_]
            sequence = (records_dict[gene_ID]).seq

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

    pars_path = 'data/parsed_data/cofactor_expression_data.csv'

    expression_dataframe.to_csv(pars_path,index=True)

    return pars_path








    



