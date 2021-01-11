import pandas as pd
import os
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
#import logging
import json
from treelstm import childsumTreeLSTM, calculate_evaluation_orders,batch_tree_input
from torch.utils import data
import subprocess
import _pickle as pickle
import torch.utils.data as data_utils
import torch.nn.functional as F
import time
from langdetect import detect
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')
# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

         
def bertembedding(text):
    # Add the special tokens.
    marked_text = "[CLS] " + text + " [SEP]"
    #print(marked_text)
    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)
    #print(tokenized_text)
    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    #print(indexed_tokens)
    # Mark each of the 22 tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_text)
    
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
    # `encoded_layers` has shape [12 x 1 x 22 x 768]
    
    # `token_vecs` is a tensor with shape [22 x 768]
    token_vecs = encoded_layers[11][0]    
    # Calculate the average of all 22 token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)
    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(encoded_layers, dim=0)
    
    token_embeddings.size()
    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    
    token_embeddings.size()
    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1,0,2)
    
    token_embeddings.size()
    # `encoded_layers` has shape [12 x 1 x 22 x 768]    
    # `token_vecs` is a tensor with shape [22 x 768]
    token_vecs = encoded_layers[11][0]
    # Calculate the average of all 22 token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)
    sentence_embedding=sentence_embedding.tolist()
    return sentence_embedding

gbbertembedd={}# for not calculating embedding everytime
with open('gbbertembedd.bin','rb') as f:
    gbbertembedd=pickle.load(f)
def createtree(replydic,textdic,srcid):
    temp=dict()
    if srcid not in gbbertembedd.keys():
        temp['features']=bertembedding(textdic[srcid])
        gbbertembedd[srcid]=temp['features']
    else:
        temp['features']=gbbertembedd[srcid]    
    #temp['features']=[0]
    childls=[]
    #print(srcid)
    if srcid in replydic.keys():
        for i in replydic[srcid]:
            #print(i)
            childls.append(createtree(replydic,textdic,i))      
    #print(childls)
    temp['children']=childls
    return temp

def cleantext(filename,output):
    #print(filename) 
    if not os.path.exists(output):
        with open(output,'w'):pass
    subprocess.call([r"python","preprocessGensim.py",filename, output])   

def _label_node_index(node):
    global glbn
    node['index'] = glbn
    for child in node['children']:
        glbn += 1
        _label_node_index(child)


def _gather_node_attributes(node, key):
    features = [node[key]]
    for child in node['children']:
        features.extend(_gather_node_attributes(child, key))
    return features


def _gather_adjacency_list(node):
    adjacency_list = []
    for child in node['children']:
        adjacency_list.append([node['index'], child['index']])
        adjacency_list.extend(_gather_adjacency_list(child))

    return adjacency_list
#this is not a right place to set the device. device should be set inside training loop for ease if more than one GPU exist 
def convert_tree_to_tensors(tree, device=torch.device('cuda:0')):
    # Label each node with its walk order to match nodes to feature tensor indexes
    # This modifies the original tree as a side effect
    _label_node_index(tree)

    features = _gather_node_attributes(tree, 'features')
    #print(features)
    #labels = _gather_node_attributes(tree, 'labels')
    adjacency_list = _gather_adjacency_list(tree)
    #print(adjacency_list)
    node_order, edge_order = calculate_evaluation_orders(adjacency_list, len(features))
    #print(node_order)
    #print(edge_order)
    return {
        'features': torch.tensor(features, device=device, dtype=torch.float32),
        #'labels': torch.tensor(labels, device=device, dtype=torch.float32),
        'node_order': torch.tensor(node_order, device=device, dtype=torch.int64),
        'adjacency_list': torch.tensor(adjacency_list, device=device, dtype=torch.int64),
        'edge_order': torch.tensor(edge_order, device=device, dtype=torch.int64),
    }

class treeDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, trees, labels):
        'Initialization'
        self.trees = trees
        self.labels = labels

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        return convert_tree_to_tensors(self.trees[index]),self.labels[index]
    
# hyper parameters
x_size = 768
h_size = 512
num_classes=2
#dropout = 0.5
lr = 0.05
weight_decay = 1e-4
# training loop
    
tempdir='unverified_tweet_dataset' #'smalldataset',
category=['charliehebdo','germanwings-crash','ottawashooting','sydneysiege']#
cattype=['non-rumours','rumours']
sttime=time.time()
for icat in range(len(category)):
    # create the model
    modeltrlstm = childsumTreeLSTM(x_size,h_size,num_classes)
    modeltrlstm.cuda()    
    optimizer = torch.optim.Adagrad(modeltrlstm.parameters(), lr=lr,weight_decay=weight_decay)
    print(modeltrlstm)
    cattree=list()# list of trees in each category for creating dataloader
    verifls=[] # stores the verified status of each source tweet
    srcidls=[]
    for c in [temp for itemp,temp in enumerate(category) if itemp!=icat]:        
        '''
        for ct in cattype:        
            path=tempdir+'/'+c+'/'+ct
            twtsdir = os.listdir(path)         
            for twt in twtsdir:
                paths=path+'/'+twt+'/source-tweet'
                source=os.listdir(paths)
                with open(paths+"/"+source[0]) as file:
                    for i,line in enumerate(file):
                        rawstring = json.loads(line)                        
                        srctext=rawstring["text"]
                        #if(detect(srctext)!='en'):#if not english then ignore currently
                        #    print(srctext)
                        #    continue
                        srcid=rawstring["id"]
                        srcidls.append(srcid)
                        verifls.append(rawstring["user"]["verified"])
                        
                resdic=dict()# key : original tweet id, value: reply ids
                textls=[]
                textdic=dict()# key :tweet id, value: text
                textls.append([srcid,srctext])
                resdic[srcid]=list()
                pathr=path+'/'+twt+'/reactions'
                reactions=os.listdir(pathr)
                for r in reactions:
                    with open(pathr+"/"+r) as file:
                        for i,line in enumerate(file):
                            rawstring = json.loads(line)
                            sid=rawstring["id"]
                            resid=rawstring["in_reply_to_status_id"]
                            restext=rawstring["text"]
                            #if int(sid)==552790459559735296:
                            #    print("replies")
                            #    print(restext)
                            if(resid not in resdic.keys()):
                                resdic[resid]=list()
                                resdic[resid].append(sid)
                            else:
                                resdic[resid].append(sid)
                            #textdic[sid]=restext
                            textls.append([sid,restext])
                
                with open("templs.txt",'w',encoding='utf-8') as f:
                    f.writelines(x[1].replace('\n', ' ').replace('\r', '')+'\n' for x in textls)
                cleantext("templs.txt","outtempls.txt") #cleaning before finding embeddings
                
                with open('outtempls.txt','r',encoding='utf-8') as f:
                    textdic={a:b for a,b in zip([x[0] for x in textls],[x for x in f.readlines()])} #creating dictionary key: tweetid value:clean tweet text
                cattree.append(createtree(resdic,textdic,srcid))
                #print(cattree[0])
        print("total time:{0}".format(time.time()-sttime))    
        #create dataset for each category
        with open(c+'tree.bin','wb') as f:
            pickle.dump(cattree,f)
        with open(c+'verifls.bin','wb') as f:
            pickle.dump(verifls,f)
        with open(c+'srcidls.bin','wb') as f:
            pickle.dump(srcidls,f)    
        '''
        with open(c+'tree.bin','rb') as f:
            cattree=pickle.load(f)
        with open(c+'verifls.bin','rb') as f:
            verifls=pickle.load(f)
        with open(c+'srcidls.bin','rb') as f:
            srcidls=pickle.load(f)        
        print(len(cattree))
        
        #trds=treeDataset(cattree,torch.tensor(verifls))   
        #train_data_generator = data_utils.DataLoader(trds)#,collate_fn=batch_tree_input,batch_size=10)
        #print("after dataloader")
        device = torch.device('cuda:0')
        epochs = 1
        
        for epoch in range(epochs):
            predls=[]
            for i,(batch,labels) in enumerate(zip(cattree,verifls)):#train_data_generator:
                glbn=0 #node count for each tree, hence resetting
                batch=convert_tree_to_tensors(batch) #returns the dictionaries            
                logp = modeltrlstm(batch['features'],batch['node_order'],batch['adjacency_list'],batch['edge_order'])
                temp=torch.zeros(1,2,device=device)
                temp[0,:]=logp #creating tensor of size[1,2]
                #print(temp.requires_grad)
                tlabel=torch.tensor([1 if labels==1 else 0],device=device) #first is verified, second position is unverified
                loss = F.nll_loss(temp, tlabel)
                #print('loss:',loss.item(), loss.requires_grad)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pred = torch.argmax(logp, 0)
                predls.append(pred.item())
                #os.environ["PATH"] += os.pathsep + 'D:/phd/ijcai/graphviz-2.38/release/bin/'
                #make_dot(loss).render("attached", format="png")
            #print([verifls[i] for i in trn_idx])
            #print(predls)
            acc = float(torch.sum(torch.eq(torch.tensor(verifls), torch.tensor(predls)))) / len(verifls)
            print("Training :category {} | Epoch {:05d} | Acc {:.4f} |".format(c,epoch, acc))
            print( classification_report(verifls, predls,zero_division=0))
    
    c=category[icat]
    with open(c+'tree.bin','rb') as f:
        cattree=pickle.load(f)
    with open(c+'verifls.bin','rb') as f:
        verifls=pickle.load(f)
    with open(c+'srcidls.bin','rb') as f:
        srcidls=pickle.load(f)    
    print(len(cattree))      
    predls=[]
    for i,(batch,labels) in enumerate(zip(cattree,verifls)):#train_data_generator:
        glbn=0 #node count for each tree, hence resetting
        batch=convert_tree_to_tensors(batch) #returns the dictionaries            
        logp = modeltrlstm(batch['features'],batch['node_order'],batch['adjacency_list'],batch['edge_order'])
        pred = torch.argmax(logp, 0)
        predls.append(pred.item())                
    #print([verifls[i] for i in val_idx])
    #print(predls)
    acc = float(torch.sum(torch.eq(torch.tensor(verifls), torch.tensor(predls)))) / len(verifls)
    print("Validation: category {} | Acc {:.4f} |".format( c, acc))
    print(classification_report(verifls, predls,zero_division=0))
    
#storing bert embedding 
#with open('gbbertembedd.bin','wb') as f:
#    pickle.dump(gbbertembedd,f)
            
