import numpy as np
import re
from random import shuffle
import time
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




# def batch_iter(data, batch_size, num_epochs, shuffle=True):
#     """
#     Generates a batch iterator for a dataset.
#     """
#     data = np.array(data)
#     data_size = len(data)
#     num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
#     for epoch in range(num_epochs):
#         # Shuffle the data at each epoch
#         if shuffle:
#             shuffle_indices = np.random.permutation(np.arange(data_size))
#             shuffled_data = data[shuffle_indices]
#         else:
#             shuffled_data = data
#         for batch_num in range(num_batches_per_epoch):
#             start_index = batch_num * batch_size
#             end_index = min((batch_num + 1) * batch_size, data_size)
#             yield shuffled_data[start_index:end_index]



#
print str(int(time.time()))
