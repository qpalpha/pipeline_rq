# -*- coding: utf-8 -*-
"""
# Author: Li Xiang@CICC.EQ
# Created Time : Mon 22 Jul 2019 01:26:03 PM CST
# File Name: test.py
# Description:
"""
#%% Import Part
from hfdata import *

#%% Test Codes
if __name__=='__main__':
    tick = TickData('./ini/snapshot.ini')
    tick.save_snapshot()
    tick.save_snapshot(all_ids()[:10])
