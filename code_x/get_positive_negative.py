import random
positive_pairs_path = "../data/lnc_mir_pairs.txt"
positive_pairs=[]
lncrna_list = []
mirna_list = []
for pair in open(positive_pairs_path,"r").readlines():
    lncrna=pair.strip().split(",")[0]
    mirna=pair.strip().split(",")[1]
    lncrna_list.append(lncrna)
    mirna_list.append(mirna)
    positive_pairs.append((lncrna,mirna))

netative_pairs=[]
while len(netative_pairs)<len(positive_pairs):
    lncrna = random.choice(lncrna_list)
    mirna = random.choice(mirna_list)
    if (lncrna,mirna) not in positive_pairs and (lncrna,mirna) not in netative_pairs:
        netative_pairs.append((lncrna,mirna))

f = open("./data/negative_pairs.txt","w")
for x in netative_pairs:
    lncrna = x[0]
    mirna = x[1]
    f.write(lncrna+","+mirna+"\n")
f.close()

