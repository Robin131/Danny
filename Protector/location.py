from datasets.Protector_location import data as PL_data
from datasets.IE import data as IE_data
from datasets import data_utils
from Model import seq2seq_wrapper
from Protector.main import output_effect, show_profile
import importlib
import time
import sys

def conversation_predict_model():
    importlib.reload(PL_data)

    # load data from pickle and npy files
    metadata, idx_q, idx_a = PL_data.load_data(PATH='../datasets/Protector_location/')
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
                                   ckpt_path='../ckpt/Protector_location/',
                                   loss_path='',
                                   metadata=metadata,
                                   emb_dim=emb_dim,
                                   num_layers=3
                                   )

    sess = model.restore_last_session()
    return model, sess, metadata

def IE_predict_model():
    importlib.reload(IE_data)

    # load data from pickle and npy files
    metadata, idx_q, idx_a = IE_data.load_data(PATH='../datasets/Protector_location_IE/')
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
                                    ckpt_path='../ckpt/Protector_location_IE/',
                                    loss_path='',
                                    metadata=metadata,
                                    emb_dim=emb_dim,
                                    num_layers=3
                                    )

    sess = model.restore_last_session()

    return model, sess, metadata

def raise_question(step):
    if step == 'hello':
        return 'Hi John. Nice to meet you?'
    elif step == 'invite':
        return 'Have you learned about the welcome party from Joseph?'
    elif step == 'help':
        return 'btw are you local? Because I met some trouble recently, and it’s just too awkward to ask help from strangers.'
    elif step == 'community':
        return 'Can I ask you something about our community?'
    elif step == 'supermarket':
        return 'Could you recommend me a supermarket? The one you often go as a local resident.'
    elif step == 's_address':
        return 'May I have the address please? Then I can search it on Google.'
    elif step == 'restaurant':
        return 'And where can I eat with my friend? I mean, Can you recommend a good restaurant for me? Maybe the one you always go to.'
    elif step == 'r_address':
        return 'Can I have the address please.'
    elif step == 'coffee_house':
        return 'And is there any good coffee shop around?'
    elif step == 'c_address':
        return 'Can you give me the address?'
    elif step == 'bus_line':
        return 'I hope the party location is not far from your living apartment because it is snowing these days.'
    elif step == 'bye':
        return 'Then see you at the party'
    elif step == 'END':
        return ''
    else:
        print('no such step, sorry')


if __name__ == '__main__':

    university = 'University of Montreal'

    dic = {
        'restaurant': '',
        'r_address': '',
        'supermarket': '',
        's_address': '',
        'coffee house': '',
        'c_address': '',
        'bus': ''
    }

    next_step = {
        'hello': 'invite',
        'invite': 'help',
        'help': 'community',
        'community': 'supermarket',
        'supermarket': 's_address',
        's_address': 'restaurant',
        'restaurant': 'r_address',
        'r_address': 'coffee_house',
        'coffee_house': 'c_address',
        'c_address': 'bus_line',
        'bus_line': 'bye',
        'bye': 'END'
    }

    IE_model, IE_sess, IE_metadata = IE_predict_model()
    conv_model, conv_sess, conv_metadata = conversation_predict_model()

    step = 'hello'
    print()

    while not step == 'END':
            question = raise_question(step)
            output_effect(question)
            input_txt = input()
            input_P = PL_data.split_sentence(input_txt, conv_metadata)
            input_IE = IE_data.split_sentence(input_txt, IE_metadata)
            input__P = input_P.T
            input__IE = input_IE.T
            if step in list(dic.keys()):
                information = IE_model.predict(IE_sess, input__IE)
                privacy = data_utils.decode(sequence=information[0], lookup=IE_metadata['idx2w'], separator=' ')
                print(privacy)
                dic[step] = privacy
            output_ = conv_model.predict(conv_sess, input__P)
            answer = data_utils.decode(sequence=output_[0], lookup=conv_metadata['idx2w'], separator=' ')

            if not ('#' in answer or len(answer) == 0):
                output_effect(answer)

            if isinstance(next_step[step], dict):
                if privacy in list(next_step[step].keys()):
                    step = next_step[step][privacy]
                else:
                    d = next_step[step]
                    step = d[list(d.keys())[0]]
            else:
                step = next_step[step]

    show_profile(dic)