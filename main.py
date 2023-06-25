from src.Parser.parser import parse_bed_file
from Bio import SeqIO
from Bio.Seq import Seq


bed_path= '/Users/ugur_dura/Desktop/IN2393-Machine Learning for Regulatory Genomics/Project/promoter_sequence_identification/Trial_Scripts/Flybase_dm6_TSSs.bed'
fasta_path = '/Users/ugur_dura/Desktop/IN2393-Machine Learning for Regulatory Genomics/Project/promoter_sequence_identification/Trial_Scripts/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fa'

records = SeqIO.to_dict(SeqIO.parse(open(fasta_path), 'fasta'))





arguments = {}

arguments["bed_path"] = bed_path
arguments["fasta_path"] = fasta_path
arguments["records"] = records
arguments["show_sequence_legth"] = True

arguments["sequence_length"] = 249







#########################################################################################################################################################

                                                        #Option 1: Run with Arguments

#########################################################################################################################################################





print("###############################   Parser is Started   ###############################")



parse_bed_file(**arguments)



print("###############################   Parser is Finalized  ###############################")