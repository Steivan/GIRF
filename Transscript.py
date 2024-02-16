# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 23:04:30 2024

@author: steiv

copied from:
https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting

and slightly modified

"""

"""
Transcript - direct print output to a file, in addition to terminal.

Usage:
    from Transcript import ts_start, ts_close
    ts_start('logfile.log')
    print("inside file")
    ts_stop()
    print("outside file")
"""

import sys

class Transcript(object):

    def __init__(self, filename, wa='w'):
        self.terminal = sys.stdout
        self.logfile = open(filename, wa)

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def ts_start(filename, wa='w'):
    """Start transcript, appending print output to given filename"""
    sys.stdout = Transcript(filename, wa)

def ts_stop():
    """Stop transcript and return print functionality to normal"""
    sys.stdout.logfile.close()
    sys.stdout = sys.stdout.terminal
    
if __name__ == "__main__":

    file_name = 'transscript_test.txt'
    
    print('Text A written to console')
    ts_start(file_name, wa='w')                       # overwrite
    print('Text B written to file and console')
    ts_stop()
    print('Text C written to console')
    ts_start(file_name, wa='a')                       # append
    print('Text D appended to file and written to console')
    ts_stop()
    print('Text E written to console')
    
      