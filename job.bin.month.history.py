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
        edate = sys.argv[1]
    except:
        edate = None
    for minute in ['5','15','30']:
        mb = MBData('./ini/mb.ini',minute)
        mb.to_month_bin(edate=edate)
