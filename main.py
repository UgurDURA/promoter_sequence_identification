from src.Parser.bed_parser import parse_bed_file
from src.Parser.cofactor_expression_parser import cofactor_expression_parser
from src.Parser.parser_helpers import get_keys, get_keys_dm3
from src.DataLoader.data_loader import bed_dataLoader, cofactor_dataLoader
from src.Network.bed_model import bed_DeepSTARR, bed_train_model
from src.Network.cofactor_model import cofactor_DeepSTARR, cofactor_train_model
from Bio import SeqIO

import torch
import torch.nn as nn
import torch.optim as optim




#########################################################################################################################################################

                                                        #Step 1: Parse the provided data

#########################################################################################################################################################






bed_path= 'data/raw_data/Flybase_dm6_TSSs.bed'
dm6_fasta_path = 'data/raw_data/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fa'
dm3_fasta_path = 'data/raw_data/BDGP_R5_dm3.fa'
expression_path = 'data/raw_data/Haberle_COF_STAP_oligos.xlsx'
output_path = 'data/parsed_data/'



records = SeqIO.to_dict(SeqIO.parse(open(dm6_fasta_path), 'fasta'))
values=[(key,value) for key, value in records.items() if 'NC' in key or 'NT' in key]
records_dict = dict(values)
chromosome_dict = get_keys(dm6_fasta_path)


records_dm3 = SeqIO.to_dict(SeqIO.parse(open(dm3_fasta_path), 'fasta'))
values_dm3=[(key,value) for key, value in records_dm3.items()]
records_dict_dm3 = dict(values_dm3)
chromosome_dict_dm3 = get_keys_dm3(dm3_fasta_path)



arguments = {}

arguments["bed_path"] = bed_path
arguments['expression_path'] = expression_path

arguments["dm6_fasta_path"] = dm6_fasta_path
arguments["dm3_fasta_path"] = dm3_fasta_path

arguments["records_dict"] = records_dict
arguments["show_sequence_legth"] = True
arguments["chromosome_dict"] = chromosome_dict

arguments["records_dict_dm3"] = records_dict_dm3
arguments["show_sequence_legth"] = True
arguments["chromosome_dict_dm3"] = chromosome_dict_dm3

arguments["sequence_length"] = 249

arguments['output_path'] = output_path




print("###############################   Parser is Started   ###############################")




# parsed_bed_file_path = parse_bed_file(**arguments)
# parsed_cofactor_path = cofactor_expression_parser(**arguments)

parsed_bed_file_path = output_path + 'parsed_bed_data.csv'
parsed_cofactor_path = output_path + 'parsed_cofactor_expression_data.csv'


print("###############################   Parser is Finalized  ###############################")




#########################################################################################################################################################

                                                        #Step 2: Load Data and Split 

#########################################################################################################################################################


hparams =             {'batch_size_train':256,#64, # number of examples per batch
                      'batch_size_vt':256,
                      'epochs': 100, # number of epochs SHOULD BE 100
                      #'early_stop': 10, # patience of 10 epochs to reduce training time; you can increase the patience to see if the model improves after more epochs
                      'lr': 0.001, # learning rate
                      #'n_conv_layer': 3, # number of convolutional layers
                      'num_filters1': 128, # number of filters/kernels in the first conv layer
                      'num_filters2': 60, # number of filters/kernels in the second conv layer
                      'num_filters3': 60, # number of filters/kernels in the third conv layer
                      'num_filters4': 120,
                      'kernel_size1': 7, # size of the filters in the first conv layer
                      'kernel_size2': 3, # size of the filters in the second conv layer
                      'kernel_size3': 5, # size of the filters in the third conv layer
                      'kernel_size4': 3,
                      'n_dense_layer': 1, # number of dense/fully connected layers
                      'dense_neurons1': 64, # number of neurons in the dense layer
                      'dense_neurons2': 256,
                      'dropout_prob': 0.4, # dropout probability
                      }




device = torch.device("mps")  # ------->>> If you are using M1/M2 (Arm) based architecture)
print('You are using the following device: ', device)


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # ---->> If you are using Intel (x64) based architecture)



                                                        #Model 1: TSS Prediction
#########################################################################################################################################################

bed_train_dataloader, bed_val_dataloader, bed_test_dataloader = bed_dataLoader(parsed_bed_file_path, hparams)

model = bed_DeepSTARR(hparams)
bed_train_model(model.to(device), bed_train_dataloader, bed_val_dataloader)


#########################################################################################################################################################




                                                        #Model 2: Cofactor Prediction
#########################################################################################################################################################

# cofactor_train_dataloader, cofactor_val_dataloader, cofactor_test_dataloader = cofactor_dataLoader(parsed_cofactor_path, hparams )

# model = cofactor_DeepSTARR(hparams)
# cofactor_train_model(model.to(device), cofactor_train_dataloader, cofactor_val_dataloader, cofactor_test_dataloader)


#########################################################################################################################################################