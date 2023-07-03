from src.Parser.bed_parser import parse_bed_file
from src.Parser.cofactor_expression_parser import cofactor_expression_parser
from src.Parser.parser_helpers import get_keys
from Bio import SeqIO
from Bio.Seq import Seq


bed_path= '/Users/ugur_dura/Desktop/IN2393-Machine Learning for Regulatory Genomics/Project/promoter_sequence_identification/Trial_Scripts/Flybase_dm6_TSSs.bed'
fasta_path = '/Users/ugur_dura/Desktop/IN2393-Machine Learning for Regulatory Genomics/Project/promoter_sequence_identification/Trial_Scripts/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fa'
expression_path = 'data/raw_data/Haberle_COF_STAP_oligos.xlsx'
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



parse_bed_file(**arguments)
cofactor_expression_parser(**arguments)



print("###############################   Parser is Finalized  ###############################")