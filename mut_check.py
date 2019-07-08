import pandas as pd
import os
import time

dataset_file_name = "pdb_mut_ddg_uniprotid.csv"
mutcheck_file_name = "mutcheck_outnew"
dataset = pd.read_csv(dataset_file_name, sep=",", header=0)
# print(mutcheck["name"][0])
# ofile = open("frag_table.txt","w+")
n = 0
df_new = pd.DataFrame(columns=['header', '+', ".", ":", "*", "ddG"])


print("start: ", time.ctime())


def _alncheck(fastafilename):
    os.system("muscle -in " + fastafilename + " -out " + fastafilename + ".aln -maxiters 16 -clw -quiet")
    alnfilename = fastafilename + ".aln"
    try:
        oo = open(alnfilename)
        frag = ""
        mark = ""
        for line in oo:
            if line.startswith(" "):
                mark = mark + line[23:].replace("\n", "")
            if line.startswith(alnfilename.replace(".fasta.aln", "")):
                frag = frag + line[23:].replace("\n", "")
        n = 1
        i = 0
        frag_dic = {}
        mark_dic = {}
        for x in frag:
            if x == "-":
                i = i + 1
            else:
                frag_dic[n] = x
                mark_dic[n] = mark[i]
                i = i + 1
                n = n + 1
        # print(frag_dic,mark_dic)
        x = alnfilename
        p = int(x.split("_")[2][1:][:-1])
        s = int(x.split("_")[3])
        m = mark_dic.get(p - s + 1)
        if m == " ":
            m = "+"
        return m
    except FileNotFoundError:
        print(alnfilename + " not found!")


def _readseq(fasta):
    seq = {}
    for line in fasta:
        if line.startswith(">"):
            header = line.replace('>', '').replace("\n", "").split(" ")[0].upper()
            seq[header] = ""
        else:
            seq[header] += line.replace("\n", "")
    return seq


def _mut(apa, sequence):  # apa for aa-position-aa
    seq_list = list(sequence)
    p = int(apa[1:][:-1])
    s = 0
    variant = ""
    for aa in seq_list:
        if s + 1 == p:
            variant = variant + apa[-1]
            s = s + 1
        else:
            variant = variant + aa
            s = s + 1
    return variant


dataset_file_name = "pdb_mut_ddg_uniprotid.csv"
dataset = pd.read_csv(dataset_file_name, sep=",", header=0)
pdb_dic = {}
hotdb_path = "/home/md/GT_60/GT_60_reduced.fastadb"
hotdb = open(hotdb_path)
mut_ddg_dic = {}
the_out_file = open("mutcheck_out", "w+")
hotdb = _readseq(hotdb)

pdbseq = open("/gdata/databases/pdb/pdb_seq.fasta")
pdbseq_dic = _readseq(pdbseq)
qualified_seq_filename = "dataset_mut.out"
qualified_seq_dataset = pd.read_csv(qualified_seq_filename, sep="\t", header=None)  # 0:query,1:best_hit
frag_list = []
for ids in qualified_seq_dataset[0]:
    position = ids.split("_")[2][1:][:-1]
    apa = ids.split("_")[2]
    head = ids[0:6]
    seq = pdbseq_dic.get(head)
    mut = _mut(apa, seq)
    n = 20   #the length of fragment
    i = 0
    fg_dic = {}
    while i < n:
        start = int(position) - n + i
        end = int(position) + i
        f = mut[start:end]
        headers = ids + "_" + str(start + 1) + "_" + str(end)
        # print(headers)
        fg_dic[headers] = f
        i = i + 1
    for key in fg_dic:
        if "-" in key:
            continue
        if "0:19" in key:
            continue
        else:
            frag_list.append(key + ".fasta")

print("frag_list built: " + time.ctime())

evalue_dic = {}
identity_dic = {}
n = 1 #the number of best hits
for x in frag_list:
    blast = os.popen(
        "blastp -query " + x + " -db  ~/GT_60/GT_60_reduced.fastadb -outfmt 6 -num_threads 50 -max_target_seqs " + str(
            n))
    blast = blast.read()
    hotid = []
    i = 0
    oseq = open(x, "a+")
    while i < n:
        try:
            iline = blast.split("\n")[i]
            hid = iline.split("\t")[1]
            identity = iline.split("\t")[2]
            identity_dic[x] = str(identity)
            evalue = iline.split("\t")[10]
            evalue_dic[x] = str(evalue)
            hotid.append(hid)
            i = i + 1
            # for hotids in hotid:
            # hotseq = hotdb[hotids]
            # print(">" + hotids + "\n" + hotseq, file=oseq)
        except IndexError:
            print(x + " Only " + str(i) + " good match(es)")
            break
    # print(hotid)
    for hotids in hotid:
        hotseq = hotdb.get(hotids)
        if hotseq == None:
            continue
        else:
            print(">" + hotids + "\n" + hotseq, file=oseq)

the_out_file = open("mutcheck_outnew", "w+")


dataset_file_name = "pdb_mut_ddg_uniprotid.csv"
mutcheck_file_name = "mutcheck_outnew"
dataset = pd.read_csv(dataset_file_name, sep=",", header=0)
mutcheck = pd.read_table(mutcheck_file_name, sep="\t", header=0)

the_out_file = open("mutcheck_outnew", "w+")
print("name" + "\t" + "mark", file=the_out_file)
the_out_file = open("mutcheck_outnew", "a+")
outline = ""
for x in frag_list:
    c = _alncheck(x)
    name = x[:-6]
    try:
        outline = outline + name + "\t" + c + "\n"
    except TypeError:
        continue
print(outline, file=the_out_file)

print("end: " + time.ctime())

mutcheck = pd.read_table(mutcheck_file_name, sep="\t", header=0)
for x in mutcheck["name"]:
    headers = x
    s = headers.split("_")
    h = s[0] + "_" + s[1] + "_" + s[2]
    marks = mutcheck["mark"][n]
    df_new = df_new.append(pd.DataFrame({'header': [h], marks: [1]}), ignore_index=True, sort=False)
    n = n + 1
    df_new = df_new.fillna(0)
    df_sum = df_new.groupby(by=['header']).sum()

mut_ddg_dic = {}
n = 0
for header in dataset["seq"]:
    h = header.lower().replace(" ", "")
    apa = dataset["Variant in PDB"][n]
    code = dataset["seq"][n]
    keys = code + "_" + apa
    mut_ddg_dic[keys.replace(" ", "")] = str(dataset["ddG"][n])
    n = n + 1

for k in mut_ddg_dic:
    df_sum.loc[k, "ddG"] = mut_ddg_dic[k]

df_sum.dropna(axis=0, how='any').to_csv("frag_table.txt")
