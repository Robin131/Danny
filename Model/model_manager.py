from datasets.danny import data as d_data
from datasets.IE import data as IE_data
import data_utils
import seq2seq_wrapper
import importlib

def get_model():
    importlib.reload(d_data)
    importlib.reload(IE_data)

    d_metadata, d_idx_q, d_idx_a = d_data.load_data(PATH='../datasets/danny/')
    i_metadata, i_idx_q, i_idx_a = IE_data.load_data(PATH='../datasets/IE/')

    (d_trainX, d_trainY), (d_testX, d_testY), (d_validX, d_validY) = data_utils.split_dataset(d_idx_q, d_idx_a)
    (i_trainX, i_trainY), (i_testX, i_testY), (i_validX, i_validY) = data_utils.split_dataset(i_idx_q, i_idx_a)

    d_model = seq2seq_wrapper.Seq2Seq(
        xseq_len=d_trainX.shape[-1],
        yseq_len=d_trainY.shape[-1],
        xvocab_size=len(d_metadata['idx2w']),
        yvocab_size=len(d_metadata['idx2w']),
        ckpt_path='../ckpt/danny/',
        loss_path='',
        metadata=d_metadata,
        emb_dim=1024,
        num_layers=3
    )

    i_model = seq2seq_wrapper.Seq2Seq(
        xseq_len=i_trainX.shape[-1],
        yseq_len=i_trainY.shape[-1],
        xvocab_size=len(i_metadata['idx2w']),
        yvocab_size=len(i_metadata['idx2w']),
        ckpt_path='../ckpt/danny/',
        loss_path='',
        metadata=i_metadata,
        emb_dim=1024,
        num_layers=3
    )

    d_sess = d_model.restore_last_session()
    i_sess = i_model.restore_last_session()

    return d_model, i_model, d_sess, i_sess, d_metadata, i_metadata

if __name__ == '__main__':
    dm, im, ds, i_s, dmt, imt = get_model()

    txt = 'I like to read'
    d_q = d_data.split_sentence(txt, dm)
    input_ = d_q.T
    output_ = dm.predict(ds, input_)
    answer = data_utils.decode(sequence=output_[0], lookup=dmt['idx2w'], separator=' ')
    print(answer)

