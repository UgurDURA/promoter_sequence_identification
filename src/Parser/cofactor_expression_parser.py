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


from src.Parser.parser_helpers import cofactor_name_exractor,get_keys

def cofactor_expression_parser(expression_path, fasta_path):

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


    
    ### Need to be discussed with Monika 
    descriptions['chrM'] = 'NC_024511.2'
    descriptions['chrU']= 'NW_007931084.1'
    descriptions['chrUextra']= 'NW_007931084.1'





    



