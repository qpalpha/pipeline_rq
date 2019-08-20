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
import pdb

#%% Test Codes
if __name__=='__main__':
    edate = today()
    # tick
    tick = TickData('./ini/tick.history.ini')
    sdate = date_offset(edate,-5)
    tick.get_raw_csv(sdate,edate)
    # mb1
    tick = TickData('./ini/mb1.history.ini')
    sdate = date_offset(edate,-3)
    tick.tick2mb1(sdate,edate)
