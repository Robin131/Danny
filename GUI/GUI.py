from tkinter import *
import time
from Model import model_manager, Q_generator
from datasets.danny import data as d_data
from datasets.IE import data as i_data
from datasets import data_utils

class ChatGUI():
    def __init__(self,
                 danny_model,
                 ie_model,
                 danny_sess,
                 ie_sess,
                 danny_metadata,
                 ie_metadata
                 ):
        self.tk = Tk()
        self.tk.title('Danny')
        self.danny_model = danny_model
        self.ie_model = ie_model
        self.danny_sess = danny_sess
        self.ie_sess = ie_sess
        self.danny_meta = danny_metadata
        self.ie_meta = ie_metadata

        self.last_reply = ''
        self.last_question = ''
        self.prev_topics = []
        self.need_reply = True
        self.human_info = {}
        self.turn = 0            # 为了按顺序提问题而临时维护的变量

        '''areas'''
        self.f_msglist = Frame(height=300, width=300)  # message_show list area
        self.f_msgsend = Frame(height=300, width=300)  # message_send list area
        f_floor = Frame(height=100, width=300)  # button area
        f_right = Frame(height=700, width=100)  # profile area

        '''control components'''
        self.txt_msglist = Text(self.f_msglist)  # txt for message_show list
        self.txt_msglist.tag_config('green', foreground='blue')  # 消息列表分区中创建标签
        self.txt_msgsend = Text(self.f_msgsend)  # txt for message_send list
        self.txt_msgsend.bind('<KeyPress-Up>', self.msgsendEvent)  # 发送消息分区中，绑定‘UP’键与消息发送。
        self.button_send = Button(f_floor, text='Send', command=self.msgsend)  # 按钮分区中创建按钮并绑定发送消息函数
        self.button_cancel = Button(f_floor, text='Cancel', command=self.cancel)  # 分区中创建取消按钮并绑定取消函数

        '''分区布局'''
        self.f_msglist.grid(row=0, column=0)  # 消息列表分区
        self.f_msgsend.grid(row=1, column=0)  # 发送消息分区
        f_floor.grid(row=2, column=0)  # 按钮分区
        f_right.grid(row=0, column=1, rowspan=3)  # 图片显示分区
        self.txt_msglist.grid()  # 消息列表文本控件加载
        self.txt_msgsend.grid()  # 消息发送文本控件加载
        self.button_send.grid(row=0, column=0, sticky=W)  # 发送按钮控件加载
        self.button_cancel.grid(row=0, column=1, sticky=W)  # 取消按钮控件加载

    def cancel(self):
        self.txt_msgsend.delete('0.0', END)
        return

    def msgsend(self):
        msg = 'H : '
        self.txt_msglist.insert(END, msg, 'green')
        self.txt_msglist.insert(END, self.txt_msgsend.get('0.0', END))
        self.last_reply = self.txt_msgsend.get('0.0', END)
        self.txt_msgsend.delete('0.0', END)
        if self.need_reply:
            msg = self.get_reply(self.last_reply)
            question, turn = Q_generator.generate_Q(self.last_reply, self.turn)
            self.turn = turn + 1
            print(question) # TODO
            self.last_question = question[1]
            self.prev_topics.append(question[1])
            self.msg_reply(msg)
            print(self.last_question)   # TODO
            if self.last_question == '':
                self.need_reply = not self.need_reply
            else:
                self.msg_reply(question[0])
        else:
            answer = self.extract_info(self.last_reply)
            self.human_info[self.prev_topics[-1]] = answer
            print(self.human_info)

        self.need_reply = not self.need_reply
        print(self.need_reply)
        return

    '''
        Send mesasge by user
    '''
    def msgsendEvent(self, event):
        if event.keysym == 'Up':
            self.msgsend()

    '''
        get a reply from Danny and return
    '''
    def get_reply(self, question):
        question = d_data.split_sentence(question, self.danny_meta)
        input_ = question.T
        output_ = self.danny_model.predict(self.danny_sess, input_)
        answer = data_utils.decode(sequence=output_[0], lookup=self.danny_meta['idx2w'], separator=' ')
        return answer

    '''
        show msg in message list
    '''
    def msg_reply(self, msg):
        msg_time = 'R : '
        self.txt_msglist.insert(END, msg_time, 'green')
        self.txt_msglist.insert(END, msg + '\n')
        self.txt_msglist.see(END)
        return

    '''
        Extract information from huamn reply
    '''
    def extract_info(self, reply):
        reply = i_data.split_sentence(reply, self.ie_meta)
        input_= reply.T
        output_ = self.ie_model.predict(self.ie_sess, input_)
        answer = data_utils.decode(sequence=output_[0], lookup=self.ie_meta['idx2w'], separator=' ')
        return answer


    def run(self):
        self.tk.mainloop()

if __name__ == '__main__':
    d_model, i_model, d_sess, i_sess, d_metadata, i_metadata = model_manager.get_model()
    window = ChatGUI(
        danny_model=d_model,
        ie_model=i_model,
        danny_sess=d_sess,
        ie_sess=i_sess,
        danny_metadata=d_metadata,
        ie_metadata=i_metadata
    )
    window.run()

