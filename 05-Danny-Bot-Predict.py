# preprocessed data
from datasets.danny import data
import data_utils
import numpy as np

import importlib
importlib.reload(data)

# load data from pickle and npy files
metadata, idx_q, idx_a = data.load_data(PATH='datasets/danny/')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

# parameters
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 16
xvocab_size = len(metadata['idx2w'])
yvocab_size = xvocab_size
emb_dim = 1024

import seq2seq_wrapper

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/danny/',
                               emb_dim=emb_dim,
                               num_layers=3
                               )

sess = model.restore_last_session()

text = ['hi', 'how', 'are', 'you']

question = np.array(data_utils.encode(sequence=text, lookup=metadata['w2idx']))

input = question.T
output  = model.predict(sess, input)

answer = data_utils.decode(sequence=output, lookup=metadata['idx2w'], separator=' ')
print(answer)




