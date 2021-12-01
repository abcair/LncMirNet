from Bio import SeqIO
import random
import networkx as nx
import numpy as np
from multiprocessing import Pool,cpu_count
import pickle

def get_paired_lnc_mirna_index(path):
    lncrna_mirna_paired = []
    lines = [x for x in open(path,"r").readlines() if "ENST" in x]
    for line in lines:
        mirna_name = line.strip().split("\t")[1]
        lncrna = line.strip().split("\t")[2]
        lncrna_names = [x for x in lncrna.split(",") if "ENST" in x]
        for lncrna_name in lncrna_names:
            if mirna_name and lncrna_name:
                lncrna_mirna_paired.append((lncrna_name,mirna_name))
    return lncrna_mirna_paired

def get_mirna_lncrna_seq(paired_path,mirna_path,lncrna_path):
    lncrna_mirna_paired = get_paired_lnc_mirna_index(paired_path)

    mirna={}
    list_mirna = list(SeqIO.parse(mirna_path,format="fasta"))
    for x in list_mirna:
        id = str(x.id)
        seq = str(x.seq).replace("U","T")
        mirna[id]=seq

    mirna_f = open("./data/mirna.list","w")
    for x in set(list(mirna.keys())):
        mirna_f.write(x+"\n")

    lncrna={}
    list_lncrna = list(SeqIO.parse(lncrna_path,format="fasta"))
    for x in list_lncrna:
        id = str(x.id)
        seq = str(x.seq).replace("U","T")
        if len(seq)>200:
            lncrna[id]=seq

    lncrna_f = open("./data/lncrna.list","w")
    for x in set(list(lncrna.keys())):
        lncrna_f.write(x+"\n")

    '''
    lnc_name, mirna_name, lnc_seq, mirna_seq
    '''

    lnc_mir_pairs_f = open("./data/lnc_mir_pairs.txt","w")
    lnc_mir_pairs_id_seq=[]
    for (lnc,mir) in lncrna_mirna_paired:
        if lnc in list(lncrna.keys()) and mir in list(mirna.keys()):
            lnc_mir_pairs_id_seq.append([lnc,mir,lncrna[lnc],mirna[mir]])
            lnc_mir_pairs_f.write(lnc+","+mir+"\n")

    return lnc_mir_pairs_id_seq,lncrna,mirna



paired_path = "./data/mirnas_lncrnas_validated.txt"
mirna_path = "./data/homo.fa"
lncrna_path = "./data/gencode.v33.lncRNA_transcripts_new.fa"
get_mirna_lncrna_seq(paired_path=paired_path,mirna_path=mirna_path,lncrna_path=lncrna_path)
