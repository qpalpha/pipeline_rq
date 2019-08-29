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
    # tickers
    tickers= Tickers('./ini/mb.ini')
    tickers.diff2csv()
