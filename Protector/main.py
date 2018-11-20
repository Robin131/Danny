from datasets.Protector_personal import data
from datasets import data_utils
from Model import seq2seq_wrapper
import importlib

def predict_model():
    importlib.reload(data)

    # load data from pickle and npy files
    metadata, idx_q, idx_a = data.load_data(PATH='../datasets/Protector_personal/')
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
                                   ckpt_path='../ckpt/Protector_personal/',
                                   loss_path='',
                                   metadata=metadata,
                                   emb_dim=emb_dim,
                                   num_layers=3
                                   )

    sess = model.restore_last_session()
    return model, sess, metadata

def raise_question(step):
    if step == 'hello':
        return 'Nice to meet you'
    elif step == 'name':
        return 'I am Victor Brown,  an exchange student here, what about you?'
    elif step == 'age':
        return 'How old are you? We might be at the same age.'
    elif step == 'university':
        return 'Are you also a college student?'
    elif step == 'major':
        return 'Are you studying Computer Science? I have heard that CS is very good at your university.'
    elif step == 'invitation':
        return 'btw, we have a welcome party for exchange student in McGill University. Lots of ' \
               'local students are coming. Do you wanna come'
    elif step == 'companion_relationship':
        return 'If you wanna come, you can also bring your friend'
    elif step == 'companion_name':
        return 'What\'s your friend\'s name? Because we may make name card for everyone'
    elif step == 'love_relationship':
        return 'And you can also bring your girl friend!'
    elif step == 'lover':
        return 'Could you tell me her name please? You know, for the name card.'
    elif step == 'phone_number':
        return 'And may I have your phone number plz? In case of any change'
    else:
        return 'No question for this step!'

if __name__ == '__main__':
    dic = {
        'name': '',
        'age': '',
        'university' : '',
        'major' : '',
        'other_guy' : '',
        'other_name' : '',
        'love_relation' : '',
        'lover': '',
        'phone_number': ''
    }

    steps = [
        'hello',
        'name',
        'age',
        'university',
        'major',
        'invitation',
        'companion_relationship',
        'companion_name',
        'love_relationship',
        'lover',
        'phone_number'
    ]

    model, sess, metadata = predict_model()

    for step in steps:
        question = raise_question(step)
        print(question)
        input_txt = input()
        input_ = data.split_sentence(input_txt, metadata)
        input__ = input_.T
        output_ = model.predict(sess, input__)
        answer = data_utils.decode(sequence=output_[0], lookup=metadata['idx2w'], separator=' ')
        print(answer)
