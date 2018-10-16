# preprocessed data
from datasets.danny import data
import data_utils

# load data from pickle and npy files
metadata, idx_q, idx_a = data.load_data(PATH='datasets/IE/')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)


# parameters
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 64
xvocab_size = len(metadata['idx2w'])
yvocab_size = xvocab_size
emb_dim = 1024


import seq2seq_wrapper

# In[7]:

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/IE/',
                               loss_path='',
                               metadata=metadata,
                               emb_dim=emb_dim,
                               num_layers=3,
                               epochs=2001
                               )


# In[8]:

val_batch_gen = data_utils.rand_batch_gen(validX, validY, 16)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)


# In[9]:
#sess = model.restore_last_session()
sess = model.train(train_batch_gen, val_batch_gen)
