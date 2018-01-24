import torch

import numpy as np
from tensorflow.contrib import learn
import torch
import torch.utils.data as Data
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from random import shuffle
import time
from nltk.stem.porter import *
import numpy as np
from tensorflow.contrib import learn
import random

print 'loading data...'
pos_data_f=open('../data/rt_polarity.pos','r')
neg_data_f=open('../data/rt_polarity.neg','r')
pos_data=pos_data_f.readlines()
neg_data=neg_data_f.readlines()

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()



def load_data_and_label(posFile,negFile):
    fp=open(posFile,'r')
    fn=open(negFile,'r')
    pos_=fp.readlines()
    neg_=fn.readlines()
    # print 'pos[0]',pos_[0]
    pos=[p.strip() for p in pos_]
    neg=[n.strip() for n in neg_]
    X=[clean_str(p)for p in pos]
    # print 'X[0]',X[0]
    X.extend([clean_str(n)for n in neg])
    y=[[0,1]for _ in pos]
    y.extend([[1,0]for _ in neg])
    fp.close()
    fn.close()
    return X,np.array(y)



def batch_iter(x,y,batchsize,num_epo):
    data=list(zip(x,y))
    num_batch_perEPO=((len(data)-1)/batchsize)+1
    for n_epo in range(num_epo):
        shuffle(data)
        for batch in range(num_batch_perEPO):
            start_idx=batch*batchsize
            yield data[start_idx:start_idx+batchsize]

pos_clean=map(clean_str,pos_data)
neg_clean=map(clean_str,neg_data)

x=pos_clean+neg_clean

stemmer = PorterStemmer()
def stemming(x):
    for idx in range(len(x)):
        sent=x[idx]
        stemmed=[stemmer.stem(w) for w in sent.strip().split()]
        x[idx]=' '.join(stemmed)
    return x
stemming(x)
y=[1 for _ in pos_clean]
y+=[0 for _ in neg_clean]
y=np.array(y)

x_text = x
max_document_length = max([len(x.split(" ")) for x in x_text])

## Create the vocabularyprocessor object, setting the max lengh of the documents.
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

## Transform the documents using the vocabulary.
x = np.array(list(vocab_processor.fit_transform(x_text)))    

## Extract word:id mapping from the object.
vocab_dict = vocab_processor.vocabulary_._mapping

## Sort the vocabulary dictionary on the basis of values(id).
## Both statements perform same task.
#sorted_vocab = sorted(vocab_dict.items(), key=operator.itemgetter(1))
sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])

## Treat the id's as index into list and create a list of words in the ascending order of id's
## word with id i goes at index i of the list.
vocabulary = list(list(zip(*sorted_vocab))[0])

print type(vocabulary)
print type(x)


index=range(len(x))
random.shuffle(index)

x=x[index]
y=y[index]


# train_ratio=0.66
# train_len=int(train_ratio*len(x))
# # test_len=len(x)-train_len
# x_train,y_train=x[:train_len],y[:train_len]
# x_test,y_test=x[train_len:],y[train_len:]


EPOCH=10
BATCH_SIZE=64


print 'data done...'

class CNN_sentiment(nn.Module):
    def __init__(self,channel_in,channel_out,embedding_dim,vocab_size,dropOut_prob):
        super(CNN_sentiment,self).__init__()
        self.Ci=channel_in
        self.Co=channel_out
        self.embedding_dim=embedding_dim
        self.vocab_size=vocab_size
        self.we = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        self.conv2d1=nn.Conv2d(self.Ci,self.Co,(3,self.embedding_dim))
        self.conv2d2=nn.Conv2d(self.Ci,self.Co,(4,self.embedding_dim))
        self.conv2d3=nn.Conv2d(self.Ci,self.Co,(5,self.embedding_dim))
        
        # self.bn_input=nn.BatchNorm1d(3*self.Co,momentum=0.1)
        # self.bn_fc=nn.BatchNorm1d(fc1_out,momentum=0.1)
        
        self.fc_drop = nn.Dropout(p=dropOut_prob)
        self.FC1=nn.Linear(3*self.Co,2)
        # self.FC2=nn.Linear(fc1_out,fc2_out)
        # self.out=nn.Linear(fc2_out,2)
        
    def conv_pool(self,x,conv): #x[N,1,W,D]
        x=conv(x) #[N,Co,W,1]
        x=F.relu(x.squeeze(3))
        m=nn.MaxPool1d(x.size(2),1) 
        return m(x).squeeze(2)# [N,Co]
    
    def forward(self,input):#[N,W]
        x=self.we(input)#[N,W,D]
        x=torch.unsqueeze(x,1) #[N,1,W,D]
        
        x1=self.conv_pool(x,self.conv2d1)#[N,Co]
        x2=self.conv_pool(x,self.conv2d2)
        x3=self.conv_pool(x,self.conv2d3)
        
        x_cat=torch.cat((x1,x2,x3),1)
        output=self.FC1(self.fc_drop(x_cat))
        return output
    
    def predict(self,input):
        output=self.forward(input)
        values, indices = torch.max(output, 1)
        return indices.data.tolist()
    
    
    def evaluate(self,preds, golds):
        tp, pp, cp = 0.0, 0.0, 0.0
        for pred, gold in zip(preds, golds):
            if pred == 1:
                pp += 1
            if gold == 1:
                cp += 1
            if pred == 1 and gold == 1:
                tp += 1
        precision = tp / pp
        recall = tp / cp
        try:
            f1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            return (precision, recall, 0.0)
        return (precision, recall, f1)
    

def kFoldCV(K,x,y):
    start=0
    dataLen=x.shape[0]/K

    for i in range(K):
        print '####################################Cross Validation ',i,'######################################'
        x_test_cv=x[start:(start+dataLen),:]
        y_test_cv=y[start:(start+dataLen)]

        train_idx=np.array(range(0,start)+range(start+dataLen,x.shape[0]))
        x_train_cv=x[train_idx[:None],:]
        y_train_cv=y[train_idx[:,None]]
        y_train_cv=y_train_cv.reshape([y_train_cv.shape[0],])

        # print 'x_test_cv:{},y_test_cv:{},x_train_cv:{},y_train_cv:{}'.format(x_test_cv.shape,y_test_cv.shape,x_train_cv.shape,y_train_cv .shape)

        x_train_tensor,x_test_tensor=torch.from_numpy(x_train_cv),torch.from_numpy(x_test_cv)
        y_train_tensor,y_test_tensor=torch.from_numpy(y_train_cv),torch.from_numpy(y_test_cv)

        torch_dataset=Data.TensorDataset(data_tensor=x_train_tensor,target_tensor=y_train_tensor)
        loader=Data.DataLoader(
            dataset=torch_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )


        cnn_sentiment=CNN_sentiment(1,200,150,len(vocabulary),0.5)


        optimizer=torch.optim.Adam(cnn_sentiment.parameters(),0.001)
        loss_func=nn.CrossEntropyLoss()

        for epoch in range(EPOCH):
            print '**********EPOCH',epoch,'*************'
            for step,(x_,y_) in enumerate(loader):
                # print step
                bx=autograd.Variable(x_)
                by=autograd.Variable(y_)
                
                output=cnn_sentiment.forward(bx)
                loss=loss_func(output,by)
                # if step%50==0:
                #     print 'loss:',loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            y_predict=cnn_sentiment.predict(autograd.Variable(x_test_tensor))
            print cnn_sentiment.evaluate(y_predict,y_test_tensor.tolist())

# print 'original x:{},y:{}'.format(x.shape,y.shape)
kFoldCV(5,x,y)


      
