from datetime import datetime as dt


def print_timed(message, depth=0, start='', end='\n'):
    print(start + str(dt.now()) + ('\t' * (depth+1)) + message, end=end)
