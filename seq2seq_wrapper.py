import tensorflow as tf
import numpy as np
import sys
from datasets.danny import data
import data_utils

PRESET_DATA = ['Hi', 'Hi there', 'What\'s up', 'What is your name', 'how old are you', 'Where are you', 'What are you doing', 'where do you live',
               'are you a bot or human', 'whats your name', 'hello', 'how are you', 'where are you from']


class Seq2Seq(object):

    def __init__(self, xseq_len, yseq_len, 
            xvocab_size, yvocab_size,
            emb_dim, num_layers, ckpt_path, metadata,loss_path='',
            lr=0.0001, 
            epochs=100000, model_name='seq2seq_model'):

        # attach these arguments to self
        self.xseq_len = xseq_len
        self.yseq_len = yseq_len
        self.ckpt_path = ckpt_path
        self.loss_path = loss_path
        self.meta_data = metadata
        self.epochs = epochs
        self.model_name = model_name


        # build thy graph
        #  attach any part of the graph that needs to be exposed, to the self
        def __graph__():

            # # placeholders
            # tf.reset_default_graph()
            self.g = tf.Graph()
            with self.g.as_default():
                #  encoder inputs : list of indices of length xseq_len
                self.enc_ip = [ tf.placeholder(shape=[None,],
                                dtype=tf.int64,
                                name='ei_{}'.format(t)) for t in range(xseq_len) ]

                #  labels that represent the real outputs
                self.labels = [ tf.placeholder(shape=[None,],
                                dtype=tf.int64,
                                name='ei_{}'.format(t)) for t in range(yseq_len) ]

                #  decoder inputs : 'GO' + [ y1, y2, ... y_t-1 ]
                self.dec_ip = [ tf.zeros_like(self.enc_ip[0], dtype=tf.int64, name='GO') ] + self.labels[:-1]


                # Basic LSTM cell wrapped in Dropout Wrapper
                self.keep_prob = tf.placeholder(tf.float32)
                # define the basic cell
                basic_cell = tf.contrib.rnn.DropoutWrapper(
                        tf.contrib.rnn.BasicLSTMCell(emb_dim, state_is_tuple=True),
                        output_keep_prob=self.keep_prob)
                # stack cells together : n layered model
                stacked_lstm = tf.contrib.rnn.MultiRNNCell([basic_cell]*num_layers, state_is_tuple=True)


                # for parameter sharing between training model
                #  and testing model
                with tf.variable_scope('decoder') as scope:
                    # build the seq2seq model
                    #  inputs : encoder, decoder inputs, LSTM cell type, vocabulary sizes, embedding dimensions
                    self.decode_outputs, self.decode_states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(self.enc_ip,self.dec_ip, stacked_lstm,
                                                        xvocab_size, yvocab_size, emb_dim)
                    # share parameters
                    scope.reuse_variables()
                    # testing model, where output of previous timestep is fed as input
                    #  to the next timestep
                    self.decode_outputs_test, self.decode_states_test = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                        self.enc_ip, self.dec_ip, stacked_lstm, xvocab_size, yvocab_size,emb_dim,
                        feed_previous=True)

                # now, for training,
                #  build loss function

                # weighted loss
                #  TODO : add parameter hint
                loss_weights = [ tf.ones_like(label, dtype=tf.float32) for label in self.labels ]
                self.loss = tf.contrib.legacy_seq2seq.sequence_loss(self.decode_outputs, self.labels, loss_weights, yvocab_size)
                # train op to minimize the loss
                self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

        sys.stdout.write('<log> Building Graph ')
        # build comput graph
        __graph__()
        sys.stdout.write('</log>')



    '''
        Training and Evaluation

    '''

    # get the feed dictionary
    def get_feed(self, X, Y, keep_prob):
        feed_dict = {self.enc_ip[t]: X[t] for t in range(self.xseq_len)}
        feed_dict.update({self.labels[t]: Y[t] for t in range(self.yseq_len)})
        feed_dict[self.keep_prob] = keep_prob # dropout prob
        return feed_dict

    # run one batch for training
    def train_batch(self, sess, train_batch_gen):
        # get batches
        batchX, batchY = train_batch_gen.__next__()
        # build feed
        feed_dict = self.get_feed(batchX, batchY, keep_prob=0.5)
        _, loss_v = sess.run([self.train_op, self.loss], feed_dict)
        return _, loss_v

    def eval_step(self, sess, eval_batch_gen):
        # get batches
        batchX, batchY = eval_batch_gen.__next__()
        # build feed
        feed_dict = self.get_feed(batchX, batchY, keep_prob=1.)
        loss_v, dec_op_v = sess.run([self.loss, self.decode_outputs_test], feed_dict)
        # dec_op_v is a list; also need to transpose 0,1 indices 
        #  (interchange batch_size and timesteps dimensions
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        return loss_v, dec_op_v, batchX, batchY

    # evaluate 'num_batches' batches
    def eval_batches(self, sess, eval_batch_gen, num_batches):
        losses = []
        replies = []
        for i in range(num_batches):
            loss_v, dec_op_v, batchX, batchY = self.eval_step(sess, eval_batch_gen)
            losses.append(loss_v)
            replies.append(np.argmax(dec_op_v, axis=2))
        return np.mean(losses), replies

    # finally the train function that
    #  runs the train_op in a session
    #   evaluates on valid set periodically
    #    prints statistics
    def train(self, train_set, valid_set, sess=None ):
        
        # we need to save the model periodically
        with self.g.as_default():
            saver = tf.train.Saver()

            # if no session is given
            if not sess:
                # create a session
                sess = tf.Session()
                # init all variables
                sess.run(tf.global_variables_initializer())

            sys.stdout.write('\n<log> Training started </log>\n')
            # run M epochs
            for i in range(self.epochs):
                try:
                    self.train_batch(sess, train_set)
                    print(i)
                    if i and i% (self.epochs//10) == 0: # TODO : make this tunable by the user

                        # save model to disk
                        saver.save(sess, self.ckpt_path + self.model_name + '.ckpt', global_step=i)
                        # evaluate to get validation loss
                        val_loss, replies = self.eval_batches(sess, valid_set, 16) # TODO : and this
                        # print stats
                        print('\nModel saved to disk at iteration #{}'.format(i))
                        print('val   loss : {0:.6f}'.format(val_loss))
                        # print('val res:')
                        # print(replies)
                        sys.stdout.flush()

                        # try preset data and save
                        if not self.loss_path == '':
                            with open(self.loss_path + 'preset' + str(i) + '.txt', 'w') as f:
                                for sentence in PRESET_DATA:
                                    question = data.split_sentence(sentence, self.meta_data)
                                    input_ = question.T
                                    output_ = self.predict(sess, input_)
                                    answer = data_utils.decode(sequence=output_[0], lookup=self.meta_data['idx2w'], separator=' ')
                                    f.write(sentence)
                                    f.write('\n')
                                    f.write(answer)
                                    f.write('\n')

                except KeyboardInterrupt: # this will most definitely happen, so handle it
                    print('Interrupted by user at iteration {}'.format(i))
                    self.session = sess
                    return sess

    def restore_last_session(self):
        with self.g.as_default():
            saver = tf.train.Saver()
            # create a session
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
            sess = tf.Session(graph=self.g, config=tf.ConfigProto(gpu_options=gpu_options))
            # get checkpoint state
            ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
            # restore session
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            # return to user
        return sess

    # prediction
    def predict(self, sess, X):
        feed_dict = {self.enc_ip[t]: X[t] for t in range(self.xseq_len)}
        feed_dict[self.keep_prob] = 1.
        dec_op_v = sess.run(self.decode_outputs_test, feed_dict)
        # dec_op_v is a list; also need to transpose 0,1 indices 
        #  (interchange batch_size and timesteps dimensions
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        # print(dec_op_v.shape)
        # return the index of item with highest probability
        return np.argmax(dec_op_v, axis=2)


