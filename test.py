# -*- coding: utf-8 -*-
"""
# Author: Li Xiang@CICC.EQ
# Created Time : Mon 22 Jul 2019 01:26:03 PM CST
# File Name: test.py
# Description:
"""
#%% Import Part
import os,sys
import rqdatac as rq
rq.init()
from eq import *
import pdb

#%% Functions
def get_ids_rq():
    # sh:'XSHG',sz:'XSHE'
    #ids = [id for id in e.all_ids() if id[0] in ['0','3','6']]
    d_ids = {''}
    ids = e.all_ids()
    ids_index_dict = e.all_ashare_index_ids_sh()
    ids2 = [ids_index_dict[id] if id in ids_index_dict else id for id in ids]
    ids_market = e.ids_market(ids,sh='.XSHG',sz='.XSHE',idx='.XSHG')
    ids_rq = [id+mkt for id,mkt in zip(ids2,ids_market)]
    return ids_rq

def yyyymmdd2yyyy_mm_dd(yyyymmdd):
    return '-'.join([yyyymmdd[:4],yyyymmdd[5:6],yyyymmdd[-2:]])

def chunks(arr,size):
    sindex = np.arange(0,len(arr),size)
    eindex = np.append(sindex[1:],len(arr)-1)
    return [arr[si:ei] for si,ei in zip(sindex,eindex)]

#%% Class of MbData
class MbData(base_):
    def __init__(self,fini,freq):
        super().__init__(fini)
        # From outside
        self.ids = e.all_ids()
        self.freq = str(freq)+'m'
        self.sub_output_dir = 'mb'+str(freq)
        # From ini
        self.output_dir = self.ini.findString('OutputDir')
        self.fields = self.ini.findStringVec('Fields')
        self.start_date = self.ini.findString('StartDate')
        self.end_date = self.ini.findString('EndDate')
        self.trade_dates = e.get_dates(self.start_date,self.end_date)
        # Get ids in format of ricequant
        self.ids_rq = get_ids_rq()

    def get_price_mb(self,ids,dt,fields):
        return rq.get_price(ids,start_date=yyyymmdd2yyyy_mm_dd(dt),\
            end_date=yyyymmdd2yyyy_mm_dd(dt),fields=fields,frequency=self.freq,\
            adjust_type='none')

    def run(self):
        # Loop self.fields
        for fld in self.fields:
            # Open file
            file_name = os.path.join(self.output_dir,self.sub_output_dir,fld+'.bin')
            h5 = pd.HDFStore(file_name,'a')
            # Loop self.trade_dates
            for dt in self.trade_dates:
                if '/d'+dt in h5.keys():
                    print('{} : {} old.'.format(fld,dt))
                else:
                    print('{} : {} new.'.format(fld,dt))
                    d_raw = pd.DataFrame(self.get_price_mb(self.ids_rq,dt,fld),columns=self.ids_rq)
                    d_raw.columns = self.ids
                    d_raw.index = d_raw.index.strftime('%H%M')
                    h5['d'+dt] = d_raw
            # Close file
            h5.close()

#%% Test Codes
if __name__=='__main__':
    mb = MbData('./ini/mb.ini',freq=5)
    mb.run()
