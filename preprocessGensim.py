import subprocess
import re
import sys
import io
from nltk.corpus import stopwords
from itertools import groupby
import os
import tempfile
from itertools import islice
# input and output files
infile = sys.argv[1]
outfile = sys.argv[2]

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

stwd = set(stopwords.words('english'))  # for faster searching set  
'''tempfile.tempdir = "./"
with open(infile,'r') as f1:
    twt=f1.readlines() #not that big
    for i in range(len(twt)/5000+1):
        with tempfile.NamedTemporaryFile(mode='w+t') as temp:   
            temp.writelines(twt[i*5000:(i+1)*5000])
            with open("POStagtwt.txt","a+") as f:
                subprocess.call(["postagger/runTagger.sh","--output-format","conll", temp.name],stdout=f) #calling tokenizer and POS tagger
            #temp.close()    
'''
print("inside preprocessgensim")
with open("POStagtwt.txt","w") as f:
    subprocess.call(["bash","postagger/runTagger.sh","--output-format","conll", infile],stdout=f) #calling tokenizer and POS tagger
with io.open("POStagtwt.txt","r",encoding='utf-8') as f:
    tokens=[x.split('\t') for x in f.readlines()]
print(len(tokens))
#os.remove("POStagtwt.txt")
with open("emnlp_dict.txt","r") as f:
    slangs = {x.split('\t')[0]:x.split('\t')[1].strip('\n') for x in f.readlines()}
#col=['N',',','O','^','S','Z','V','L','M','A','R','!','D','P','&','T','X','Y','#','@','~','U','E','$','G','badwords','oov','wordratio','charcnt','stopwords']
temp=list()
with io.open(outfile, 'w',encoding='utf-8') as tweet_processed_text:
    for t in tokens:
        if(t[0]=='\n'):# new tweet starting
            tweet_processed_text.write(str(u' '.join(temp)+'\n'))
            del temp[:]
        else:
            t[0]=t[0].lower()
            t[0]=''.join(''.join(s)[:2] for _, s in groupby(t[0])) # for removing multiple occurences of characters yahooooo           
            t[0]=decontracted(t[0])
            #spelling correction should be included here
            #if t[0] in stwd: #not giving good result
            #    continue
            if t[0] in slangs:
                #print("slangs:",t[0],slangs[t[0]])
                temp.append(slangs[t[0]])
            elif((t[1]=='#' and len(t[0])>2) or (t[0][0:1]=='#' and (t[1]=='^' or t[1]=='N' or t[1]=='Z' or t[1]=='V'
            or t[1]=='!'))): #since pos tagger taking few tags as proper noun this check is here
                #print t[0][1:]
                temp.append(t[0][1:])
            elif(t[1]=='@'):
                temp.append(t[0][1:])#continue#temp.append("@user") #removing names but could relate to other tweets by name
            elif(t[1]=='U'):
                continue#temp.append("!url")    
            elif(t[0]=='rt'):
                continue#temp.append("!url")    
            elif(t[1]=='E'):
                continue
            elif(t[1]=='$'):
                temp.append(t[0])#temp.append("num")        
            elif(t[1]==','):
                temp.append(t[0])#continue
            elif(t[1]=='G'):
                continue
            elif(t[1]=='~'):   # : ... rt >>>>             
                continue
            else:
                temp.append(t[0])

