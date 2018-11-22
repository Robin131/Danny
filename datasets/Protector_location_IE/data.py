EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz# '

limit = {
        'maxq' : 25,
        'minq' : 0,
        'maxa' : 10,
        'mina' : 0
        }

UNK = 'unk'
VOCAB_SIZE = 2000

import os, re

'''
    Get sentences and targets
      sentences as questions,
      targets as answer
'''

def get_QA():
    path = './raw_data/'
    files = os.listdir(path)

    questions = []
    answers = []

    # flag = True

    for f_name in files:
        with open(path + f_name, 'r') as f:
            # print(f_name)
            lines = f.readlines()
            for line in lines:
                sentence = line.split('\t:\t')
                assert len(sentence) == 2
                questions.append(sentence[0])
                answers.append(sentence[1])
                assert len(questions) == len(answers)
    return questions, answers


'''
 remove anything that isn't in the vocabulary
    return str(pure en)

'''
def filter_line(line, whitelist):
    return ''.join([ ch for ch in line if ch in whitelist ])


'''
 filter too long and too short sequences
    return tuple( filtered_ta, filtered_en )

'''
def filter_data(qseq, aseq):
    filtered_q, filtered_a = [], []
    raw_data_len = len(qseq)

    assert len(qseq) == len(aseq)

    for i in range(raw_data_len):
        qlen, alen = len(qseq[i].split(' ')), len(aseq[i].split(' '))
        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= limit['maxa']:
                filtered_q.append(qseq[i])
                filtered_a.append(aseq[i])

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len)*100/raw_data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a


'''
 read list of words, create index to word,
  word to index dictionaries
    return tuple( vocab->(word, count), idx2w, w2idx )

'''
import nltk
import itertools

def index_(tokenized_sentences, vocab_size):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = ['_'] + [UNK] + [ x[0] for x in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist


'''
 filter based on number of unknowns (words not in vocabulary)
  filter out the worst sentences

'''
def filter_unk(qtokenized, atokenized, w2idx):
    data_len = len(qtokenized)

    filtered_q, filtered_a = [], []

    for qline, aline in zip(qtokenized, atokenized):
        unk_count_q = len([ w for w in qline if w not in w2idx ])
        unk_count_a = len([ w for w in aline if w not in w2idx ])
        if unk_count_a <= 2:
            if unk_count_q > 0:
                if unk_count_q/len(qline) > 0.2:
                    pass
            filtered_q.append(qline)
            filtered_a.append(aline)

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((data_len - filt_data_len)*100/data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a


'''
 create the final dataset : 
  - convert list of items to arrays of indices
  - add zero padding
      return ( [array_en([indices]), array_ta([indices]) )

'''
import numpy as np

def zero_pad(qtokenized, atokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'])

        # print(len(idx_q[i]), len(q_indices))
        # print(len(idx_a[i]), len(a_indices))
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a


'''
 replace words with indices in a sequence
  replace with unknown if word not in lookup
    return [list of indices]

'''

def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0] * (maxlen - len(seq))



'''
    Process data to give all the dataset and dictionary
'''

import pickle

def process_data():
    # get all questions and answers
    questions, answers = get_QA()

    # change to lower case
    questions = [ line.lower() for line in questions ]
    answers = [line.lower() for line in answers]

    # filter out unnecessary characters
    questions = [filter_line(line, EN_WHITELIST) for line in questions]
    answers = [filter_line(line, EN_WHITELIST) for line in answers]

    # filter out too long or too short sequences
    qlines, alines = filter_data(questions, answers)

    # convert questions and answers to list of words instead of sentences
    qtokenized = [[w.strip() for w in wordlist.split(' ') if w] for wordlist in qlines]
    atokenized = [[w.strip() for w in wordlist.split(' ') if w] for wordlist in alines]

    # get index for words according to distribution
    idx2w, w2idx, freq_dist = index_(qtokenized + atokenized, vocab_size=VOCAB_SIZE)

    # filter out sentences with too many unknowns
    qtokenized, atokenized = filter_unk(qtokenized, atokenized, w2idx)

    # padding sentences to fix length and replace word with index
    idx_q, idx_a = zero_pad(qtokenized, atokenized, w2idx)

    # save questions and answers file
    np.save('idx_q.npy', idx_q)
    np.save('idx_a.npy', idx_a)

    # save necessary dictionary
    metadata = {
        'w2idx': w2idx,
        'idx2w': idx2w,
        'limit': limit,
        'freq_dist': freq_dist
    }

    with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    print(len(qtokenized))

if __name__ == '__main__':
    process_data()


def load_data(PATH=''):
    # read data control dictionaries
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')
    return metadata, idx_q, idx_a

'''
    Cut a sentence into several valid words for the model
    1. filter all words not in dict
    2. check whether there is unk word
    3. cut the sentence into several words
'''
def split_sentence(sentence, metadata):
    question = [sentence]
    w2idx = metadata['w2idx']

    # fake an answer to use the previous functions
    answer = ['hi there']

    # change to lower case
    question = [line.lower() for line in question]

    # filter out unnecessary characters
    question = [filter_line(line, EN_WHITELIST) for line in question]


    # TODO if question too long, only check first 25 words

    # convert questions and answers to list of words instead of sentences
    question = [[w.strip() for w in wordlist.split(' ') if w] for wordlist in question]
    answer = [[w.strip() for w in wordlist.split(' ') if w] for wordlist in answer]

    # padding sentences to fix length and replace word with index
    idx_q, idx_a = zero_pad(question, answer, w2idx)

    return idx_q

