#!/usr/bin/env python
# coding: utf-8

# In[114]:


hydropathy_index = {"R":-2.5,"K":-1.5,"D":-0.9,"Q":-0.85,"N":-0.78,"E":-0.74,"H":-0.4,"S":-0.18,"T":
                    -0.05,"P":0.12,"Y":0.26,"C":0.29,"G":0.48,"A":0.62,"M":0.64,"W":0.81,"L":1.1,"V"
                    :1.1,"F":1.2,"I":1.4}
MW_index = {"G":75.07,"A":89.09,"V":117.15,"L":131.17,"I":131.17,"F":165.19,"W":204.23,"Y":181.19,"D"
            :133.1,"H":155.16,"N":132.12,"E":147.13,"K":146.19,"Q":146.15,"M":149.21,"R":174.2,"S":
            105.09,"T":119.12,"C":121.16,"P":115.13}
pI_index = {"G":6.06,"A":6.11,"V":6,"L":6.01,"I":6.05,"F":5.49,"W":5.89,"Y":5.64,"D":2.85,"H":7.6,"N":
            5.41,"E":3.15,"K":9.6,"Q":5.65,"M":5.74,"R":10.76,"S":5.68,"T":5.6,"C":5.05,"P":6.3}
pKa1_index = {"G":2.35,"A":2.35,"V":2.39,"L":2.33,"I":2.32,"F":2.2,"W":2.46,"Y":2.2,"D":1.99,"H":1.8,
              "N":2.14,"E":2.1,"K":2.16,"Q":2.17,"M":2.13,"R":1.82,"S":2.19,"T":2.09,"C":1.92,"P":1.95}
pKa2_index = {"G":9.78,"A":9.87,"V":9.74,"L":9.74,"I":9.76,"F":9.31,"W":9.41,"Y":9.21,"D":9.9,"H":9.33
              ,"N":8.72,"E":9.47,"K":9.06,"Q":9.13,"M":9.28,"R":8.99,"S":9.21,"T":9.1,"C":10.7,"P":10.64}
pKa3_index = {"D":3.9,"E":4.07,"H":6.04,"C":8.37,"Y":10.46,"K":10.54,"R":12.48}
#print(hydropathy_index,MW_index, pI_index, pKa1_index,pKa2_index,pKa3_index)


# In[115]:


def _precent(seq):
    dic_pseq = {}
    n = 1
    for x in seq:
        l = len(seq)
        p = n/l
        dic_pseq[p]=x
        n = n + 1
    return dic_pseq
#seq = "GENE"
#_precent(seq)


# In[116]:


import pandas as pd
def proframe(seq):
    dfname = 'df_'+str(seq)
    dfname = pd.DataFrame(columns=('p','hi','mi','ii','k1i','k2i','k3i'))
    i = 1
    for k in _precent(seq):
        a = _precent(seq).get(k)
        lst = [k,hydropathy_index.get(a),MW_index.get(a), pI_index.get(a), pKa1_index.get(a),
               pKa2_index.get(a),pKa3_index.get(a)]
        dfname.loc[i]=lst
        i = i + 1
    return dfname
#print(proframe(seq))


# In[117]:


proteins = open("/Users/jsun/RESEARCH/ML_and_protein/curated_protien.fasta")
def _readseq(fasta):
    dic = {}
    for l in fasta:
        if l.startswith(">"):
            name = l.split("|")[1]
            dic[name] = ''
        else:
            dic[name] = dic[name] + l.replace("\n","")
    return dic
proteins = _readseq(proteins)
#print(proteins)


# In[118]:


def _3_2_1(x):
    d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
         'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
    y = d.get(x)
    return y
def _curate(l):
    lst = l.split(",")
    name = lst[1]
    pos = lst[7]
    pos = int(pos)
    aa = _3_2_1(lst[5].upper())
    return name,pos,aa
newdic = {}
def seq_check(name,pos,aa):
    try:
        seq = proteins[name]
    except KeyError:
        print(name+" protein not exists")
        seq = ''
    try:
        a = seq[int(pos)-1]
    except IndexError:
        print(name+" NOT OK")
        a = ''
    if a == aa:
        newdic[name]=seq
    else:
        print(name+" NOT OK")

curated_data = open('/Users/jsun/RESEARCH/ML_and_protein/curated_residue1.csv')
data = pd.read_csv('/Users/jsun/RESEARCH/ML_and_protein/curated_residue1.csv')
#print(data.head())
next(curated_data)
for l in curated_data:
    name,pos,aa = _curate(l)
    seq_check(name,pos,aa)
#print(data.iloc[:,1])
#print(newdic)
n = 0
for k in newdic:
    n = n + 1
print(n+" records correct!")


# In[111]:



#k = proframe(newdic['P00469'])
#print(k)


# In[ ]:




