import sys
import os
from datetime import datetime

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# sys.stdout = Logger('a.txt')

if __name__ == '__main__':
    sys.stdout = Logger('a.txt')
    now = datetime.now()
    strtime = now.strftime('%b%d%H%M')
    print(strtime)
    print(now.strftime('%a, %b %d %H:%M'))
