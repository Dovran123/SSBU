from Bio import SeqIO
import random

# TODO Task 1
record = SeqIO.read("C:/laragon/www/mojAdresar2/SSBU/t10_cv9/dna_rna/inputs/sequence.gb", "genbank")
#print(record)

# TODO Task 2
dna = "5'-TACCGGAT-3'"
table = dna.maketrans("ATGC", "TACG")
#print(dna.translate(table))
# TODO Task 3
record = SeqIO.read("C:/laragon/www/mojAdresar2/SSBU/t10_cv9/dna_rna/inputs/sequence.fasta", "fasta")
#print(record)
def mutate(dna):
    dna_list  =  list(dna)
    index = random.randint(0,len(dna_list) - 1)
    dna_list[index] = random.choice("ATGC")
    return "".join(dna_list)
dna = record.seq

for i in range(1000):
    dna = mutate(dna)

#print(dna)
# TODO Task 4
print( (record.count("C") + record.count("G")) /  len(record.seq))