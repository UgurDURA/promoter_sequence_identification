
import string 
import re
import random as rnd


import time

def progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()



def sequence_len_tuner(sequence_length):

    refined_len = int((sequence_length - 1 ) / 2)

    return refined_len




def remove_dups(bedfile):
    unique_lines = set()
    with open(bedfile, "r") as f:
        for line in f:
            cleaned_line = line.strip()
            unique_lines.add(cleaned_line)
    output_file="data/parsed_data/Flybase_dm6_TSSs_cleaned.bed"
    with open(output_file, "w") as f:
        for line in unique_lines:
            f.write(line + "\n")


def reverse_complementary_sequence(dna_sequence):
    dna_sequence= dna_sequence.upper()
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    reverse_complementary_seq = ''.join(complement[base] for base in reversed(dna_sequence))
    return reverse_complementary_seq





def get_keys(fasta_path):
    pattern = r'>\s*([^ ]+)\s+.*chromosome\s+([^\s]+)'

    fasta_file = open(fasta_path, 'r')
    fasta_lines = fasta_file.readlines()

    descriptions = {}

    print("Found Fasta Keys ---->>>\n" )

    for line in fasta_lines:
        if line.startswith('>'):
            if not "sequence" in line:
                print(line)
                match = re.search(pattern, line)
                if match:
                    chromosome = "chr" + match.group(2)
                    value = match.group(1)
    
                    descriptions[chromosome] = value
    


    return descriptions





def positive_chr_dict(chromosome_dict, bed_csv, sequence_len):
    table = bed_csv
    positive_chr_dict={}
    for key in chromosome_dict.keys():
        positive_chr_dict[key]=[]
    for i in range(0,len(table)):
        if(table.iloc[i][5]=="+"):
            positive_chr_dict[table.iloc[i][0]].append(int(table.iloc[i][1])-sequence_len)
    return positive_chr_dict




def all_chr_dict(chromosome_dict,bed_csv, sequence_len):
        table = bed_csv

        all_chr_dict={}
        for key in chromosome_dict.keys():
            all_chr_dict[key]=[]
        for i in range(0,len(table)):
                all_chr_dict[table.iloc[i][0]].append(int(table.iloc[i][1])-sequence_len)
        return all_chr_dict




def create_control_seq(chr_name,positive_chr_dict,chromosome_dict, records_dict , all_chr_dict,sequence_len):

    whole_sequence_len = 2*sequence_len
    samples=[]
    num_samples=len(all_chr_dict[chr_name])*2
    for i in range(0,num_samples):
        while(True):
            control_start=rnd.randint(0,len(records_dict[chromosome_dict[chr_name]].seq)-whole_sequence_len)
            control_stop=control_start+whole_sequence_len #might be 249
            if(control_start in samples):
                continue
            for j in range(0,len(positive_chr_dict[chr_name])):
                if(((control_start>positive_chr_dict[chr_name][j]) and (control_start<positive_chr_dict[chr_name][j]+whole_sequence_len))or 
                    ((control_stop>positive_chr_dict[chr_name][j]) and (control_stop<positive_chr_dict[chr_name][j]+whole_sequence_len))):
                    break
            else:
                samples.append(control_start)
                break
    return samples

def create_control_seq_y(chromosome_dict, records_dict ,sequence_len):
    whole_sequence_len = 2*sequence_len
    samples=[]
    num_samples=3000
    for i in range(0,num_samples):
        while(True):
            control_start=rnd.randint(0,len(records_dict[chromosome_dict["chrY"]].seq)-whole_sequence_len)
            if(control_start in samples):
                continue
            samples.append(control_start)
            break
    return samples




def cofactor_name_exractor(expression_dataframe):

    columns = expression_dataframe.columns

    cofactors =columns[2:].to_list()

    return cofactors



