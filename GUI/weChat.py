import itchat
from Protector.personal_info import step as p_step
from Protector.personal_info import dic as p_dic
from Protector.personal_info import next_step as p_next
from Protector.personal_info import raise_question
from Protector.personal_info import conversation_predict_model, IE_predict_model
from datasets.Protector_personal import data as PP_data
from datasets.IE import data as IE_data
from datasets import data_utils

user_name = 'Sunshine'

step = p_step
dic = p_dic
next_step = p_next

IE_model, IE_sess, IE_metadata = None, None, None
conv_model, conv_sess, conv_metadata = None, None, None


def find_friend(nick_name):
    for friend in itchat.get_friends():
        if friend['NickName'] == nick_name:
            return friend


def get_reply(msg):
    global step, dic, next_step
    input_txt = msg
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

    if isinstance(next_step[step], dict):
        if privacy in list(next_step[step].keys()):
            step = next_step[step][privacy]
        else:
            d = next_step[step]
            step = d[list(d.keys())[0]]
    else:
        step = next_step[step]

    next_question = raise_question(step)

    return [answer, next_question]


@itchat.msg_register(itchat.content.TEXT)
def get_question(recv_msg):
    msg = recv_msg['Text']
    replies = get_reply(msg)
    reply = replies[0]
    next_question = replies[1]

    if not ('#' in reply or len(reply) == 0):
        itchat.send(msg=reply, toUserName=recv_msg['FromUserName'])
    itchat.send(msg=next_question, toUserName=recv_msg['FromUserName'])

    return

def initial():
    global IE_model, IE_sess, IE_metadata, conv_model, conv_sess, conv_metadata
    IE_model, IE_sess, IE_metadata = IE_predict_model()
    conv_model, conv_sess, conv_metadata = conversation_predict_model()


def main():
    initial()
    itchat.auto_login(True)
    greeting = raise_question(step)
    friend = find_friend(user_name)
    itchat.send(msg=greeting, toUserName=friend['UserName'])
    itchat.run()

if __name__ == '__main__':
    main()