import gensim
from Bio import SeqIO

def read_fa(path):
    res={}
    rescords = list(SeqIO.parse(path,format="fasta"))
    for x in rescords:
        id = str(x.id)
        seq = str(x.seq).replace("U","T")
        res[id]=seq
    return res

def train_doc2vec_model(seq_list,model_name):
    tokens = []
    for i, seq in enumerate(seq_list):
        items = []
        k = 0
        while k + 3 < len(seq):
            item = seq[k:k + 3]
            items.append(item)
            k = k + 1
        doc2vec_data = gensim.models.doc2vec.TaggedDocument(items, [i])
        tokens.append(doc2vec_data)
    print("-----begin train-----")
    model = gensim.models.doc2vec.Doc2Vec(vector_size=256, min_count=3, epochs=100, workers=12)
    model.build_vocab(tokens)
    model.train(tokens, total_examples=model.corpus_count, epochs=model.epochs)
    model.save("./data/"+model_name+".model")

mirna_dict = read_fa("./data/homo_mature_mirna.fa")
mirna_list = list(mirna_dict.values())
train_doc2vec_model(mirna_dict,"mirna_doc2vec")

lncrna_dict = read_fa("./data/gencode.v33.lncRNA_transcripts.fa")
lncrna_list = list(lncrna_dict.values())
train_doc2vec_model(lncrna_list,"lncrna_doc2vec")