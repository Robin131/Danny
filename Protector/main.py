from datasets.Protector_personal import data as PP_data
from datasets.IE import data as IE_data
from datasets import data_utils
from Model import seq2seq_wrapper
import importlib
import time
import sys

def conversation_predict_model():
    importlib.reload(PP_data)

    # load data from pickle and npy files
    metadata, idx_q, idx_a = PP_data.load_data(PATH='../datasets/Protector_personal/')
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

def IE_predict_model():
    importlib.reload(IE_data)

    # load data from pickle and npy files
    metadata, idx_q, idx_a = IE_data.load_data(PATH='../datasets/Protector_personal_IE/')
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

    return model, sess, metadata



def raise_question(step):
    if step == 'hello':
        return 'Nice to meet you'
    elif step == 'name':
        return 'I am Victor Brown, an exchange student here, what about you?'
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
    elif step == 'bye':
        return 'Bye'
    else:
        return 'No question for this step!'

def output_effect(txt):
    for ch in txt:
        print(ch, end='')
        sys.stdout.flush()
        time.sleep(0.1)
    print()

def phase1_context():
    txt = 'HELLO! Welcome to XXX! My name is Protector, your chatbot assistant to help your study here! \n' \
          'I will recommend you some courses on PRAVICY PROTECTION and ONLINE DECEPTION. \n' \
          'These courses will show you how you can protect yourself in present society. \n' \
          'But before this, meet with another new student and have a chat. \n' \
          'He/She might be your classmates in the following courses and you may have some exercises together. \n' \
          'Have a good chat! \n'
    return txt

def feedback_txt():
    return

def show_profile(dic):
    print()
    print('Here comes your profile:')
    for (key, value) in dic.items():
        print(key + ' : ' + value )


if __name__ == '__main__':
    dic = {
        'name': '',
        'age': '',
        'university' : '',
        'major' : '',
        'companion_relationship' : '',
        'companion_name' : '',
        'love_relationship' : '',
        'lover': '',
        'phone_number': ''
    }

    next_step = {
        'hello': 'name',
        'name': 'age',
        'age': 'university',
        'university': 'major',
        'major': 'invitation',
        'invitation': 'companion_relationship',
        'companion_relationship' : 'companion_name',
        'companion_name': 'love_relationship',
        'love_relationship': {'yes':'lover', 'none': 'phone_number'},
        'lover': 'phone_number',
        'phone_number': 'bye',
        'bye': 'END'
    }

    IE_model, IE_sess, IE_metadata = IE_predict_model()
    conv_model, conv_sess, conv_metadata = conversation_predict_model()

    phase1_txt = phase1_context()
    print()
    print(phase1_txt)

    input()

    step = 'hello'
    print()

    while not step == 'END':
            question = raise_question(step)
            output_effect(question)
            input_txt = input()
            input_P = PP_data.split_sentence(input_txt, conv_metadata)
            input_IE = IE_data.split_sentence(input_txt, IE_metadata)
            input__P = input_P.T
            input__IE = input_IE.T
            if step in list(dic.keys()):
                information = IE_model.predict(IE_sess, input__IE)
                privacy = data_utils.decode(sequence=information[0], lookup=IE_metadata['idx2w'], separator=' ')
                # print(privacy)
                dic[step] = privacy
            output_ = conv_model.predict(conv_sess, input__P)
            answer = data_utils.decode(sequence=output_[0], lookup=conv_metadata['idx2w'], separator=' ')
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