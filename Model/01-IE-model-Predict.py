# preprocessed data
from datasets.IE import data
from datasets import data_utils
from Model import seq2seq_wrapper

import importlib
importlib.reload(data)

# load data from pickle and npy files
metadata, idx_q, idx_a = data.load_data(PATH='../datasets/Protector_personal_IE/')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)



# parameters
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
xvocab_size = len(metadata['idx2w'])
yvocab_size = xvocab_size
emb_dim = 1024

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='../ckpt/PI_IE/',
                               loss_path='',
                               metadata=metadata,
                               emb_dim=emb_dim,
                               num_layers=3
                               )

sess = model.restore_last_session()
print('\nReady to test!\n')
input_txt = input()

while not input_txt == '[End]':
    question = data.split_sentence(input_txt, metadata)
    input_ = question.T
    output_ = model.predict(sess, input_)
    answer = data_utils.decode(sequence=output_[0], lookup=metadata['idx2w'], separator=' ')
    print(answer)
    input_txt = input()


