import time
import sys

def output_effect(txt):
    for ch in txt:
        print(ch, end='')
        sys.stdout.flush()
        time.sleep(0.2)

if __name__ == '__main__':
    output_effect('I am ZHANG NAN')