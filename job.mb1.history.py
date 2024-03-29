# -*- coding: utf-8 -*-
"""
# Author: Li Xiang@CICC.EQ
# Created Time : Mon 22 Jul 2019 01:26:03 PM CST
# File Name: test.py
# Description:
"""
#%% Import Part
from hfdata import *
from qpc import *
import sys
import pdb

#%% Test Codes
if __name__=='__main__':
    try:
        sdate = sys.argv[1]
    except:
        sdate = None
    try:
        edate = sys.argv[2]
    except:
        edate = None
    tick = TickData('./ini/mb1.history.ini')
    tick.tick2mb1(sdate,edate)
