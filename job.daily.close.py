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
    # tickers
    tickers= Tickers('./ini/mb.ini')
    tickers.update_and_save()
    # tick
    tick = TickData('./ini/tick.history.ini')
    sdate = date_offset(edate,-5)
    for type in ['CS','ETF','INDX','Future','Option']:
        tick.get_raw_csv(sdate,edate,type=type)
    ## mb1
    #tick = TickData('./ini/mb1.history.ini')
    #sdate = date_offset(edate,-3)
    #tick.tick2mb1(sdate,edate)
    ## mb1->mb5/15/30
    #for minute in ['5','15','30']:
    #    mb = MBData('./ini/mb.ini',minute)
    #    mb.to_csv(sdate,edate)
