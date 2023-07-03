from src.Parser.bed_parser import parse_bed_file
from src.Parser.cofactor_expression_parser import cofactor_expression_parser
from src.Parser.parser_helpers import get_keys
from src.DataLoader.data_loader import dataLoader
from src.Network.model import NeuralNetwork

from Bio import SeqIO

import torch
import torch.nn as nn
import torch.optim as optim



bed_path= '/Users/ugur_dura/Desktop/IN2393-Machine Learning for Regulatory Genomics/Project/promoter_sequence_identification/Trial_Scripts/Flybase_dm6_TSSs.bed'
fasta_path = '/Users/ugur_dura/Desktop/IN2393-Machine Learning for Regulatory Genomics/Project/promoter_sequence_identification/Trial_Scripts/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fa'
expression_path = 'data/raw_data/Haberle_COF_STAP_oligos.xlsx'

batch_size = 64





records = SeqIO.to_dict(SeqIO.parse(open(fasta_path), 'fasta'))
values=[(key,value) for key, value in records.items() if 'NC' in key or 'NT' in key]
records_dict = dict(values)
chromosome_dict = get_keys(fasta_path)




arguments = {}

arguments["bed_path"] = bed_path
arguments['expression_path'] = expression_path
arguments["fasta_path"] = fasta_path
arguments["records_dict"] = records_dict
arguments["show_sequence_legth"] = True
arguments["chromosome_dict"] = chromosome_dict

arguments["sequence_length"] = 249









#########################################################################################################################################################

                                                        #Option 1: Run with Arguments

#########################################################################################################################################################





print("###############################   Parser is Started   ###############################")



parsed_bed_file_path = parse_bed_file(**arguments)
parsed_cofactor_path = cofactor_expression_parser(**arguments)



train_dataloader, val_dataloader, test_dataloader = dataLoader(parsed_cofactor_path, batch_size)


# # Define the hyperparameters
# input_size = 784  # Example: MNIST images (28x28) flattened to 784-dimensional vectors
# hidden_size = 128
# num_classes = 10
# learning_rate = 0.001
# num_epochs = 10


# # Create an instance of the neural network
# model = NeuralNetwork(input_size, hidden_size, num_classes)

# # Define the loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# # Training loop
# for epoch in range(num_epochs):
#     model.train()  # Set the model to training mode
#     for images, labels in train_dataloader:
#         # Forward pass
#         images = images.view(-1, input_size)
#         outputs = model(images)
#         loss = criterion(outputs, labels)
        
#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
    
#     # Evaluation on validation set
#     model.eval()  # Set the model to evaluation mode
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for images, labels in val_dataloader:
#             images = images.view(-1, input_size)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
        
#         accuracy = 100 * correct / total
#         print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {accuracy:.2f}%')

# # Evaluation on test set
# model.eval()  # Set the model to evaluation mode
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_dataloader:
#         images = images.view(-1, input_size)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     accuracy = 100 * correct / total
#     print(f'Test Accuracy: {accuracy:.2f}%')









print("###############################   Parser is Finalized  ###############################")