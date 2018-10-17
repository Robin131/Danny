import os

'''
    This file is to generate training data for IE system
'''

FUNCTIONS = ['name', 'age', 'live_place', 'hobby']

'''
    Combine sentence framework with answer
'''
def combine(sentences, answer):
    res = []
    targets = []
    for (s_f, s_e) in sentences:
        for a in answer:
            res.append(s_f + a + s_e)
            targets.append(a)
    return res, targets


def name():
    sentence = [
        ('My name is ', ''),
        ('My name\'s ', ''),
        ('I am ', ''),
        ('I\'m ', ''),
        ('', '')
    ]

    name = [
        'Victor',
        'John',
        'Nan',
        'Sean'
    ]
    return sentence, name

def age():
    sentence = [
        ('I am ', ''),
        ('I am ', ' years old'),
        ('I am ', ' year old'),
        ('I\'m ', ''),
        ('I\'m ', ' years old'),
        ('I\'m ', ' year old'),
        ('', '')
    ]

    age = [str(i) for i in range(15, 30)]

    return sentence, age

def live_place():
    sentence = [
        ('', ''),
        ('I live in ', ''),
        ('I am from ', ''),
        ('I\'m from ', ''),
        ('I come from ', '')
    ]

    place = [
        'China',
        'the US',
        'US',
        'Canada',
        'Quebec',
        'Montreal'
    ]

    return sentence, place

def hobby():
    sentence = [
        ('', ''),
        ('I love ', ''),
        ('I love ', 'most'),
        ('I like to ', ''),
        ('I like to ', 'most'),
        ('I like ', ''),
        ('I like ', 'most'),
    ]

    hobby = [
        'surfing Internet',
        'cats',
        'reading',
        'playing video games'
    ]

    return sentence, hobby

if __name__ == '__main__':
    path = '../datasets/IE/raw_data/'
    for i in FUNCTIONS:
        with open(path + i + '.txt', 'w') as file:
            f = eval(i)
            sentence, target = f()
            res, targets = combine(sentence, target)
            assert  len(res) == len(targets)
            for i in range(len(res)):
                file.write(res[i] + '\t:\t' + targets[i] + '\n')
