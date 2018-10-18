'''
    This model is designed to generate questions
        for current period, we just give preset questions one by one
'''

QUESTION_LIST = [
    ('', ''),
    ('', ''),
    ('And you, how can I call you', 'name'),
    ('And what about you', 'age'),
    ('Where are you from', 'place'),
    ('What do you like to do in your spare time', 'hobby'),
    ('', '')
]

TURN = 0

'''
    Generate questions according to human reply(or question)
'''
def generate_Q(reply, turn=TURN):
    return QUESTION_LIST[turn], turn