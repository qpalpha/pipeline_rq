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
from qp import *
#from eq import *
import warnings
warnings.filterwarnings("ignore")
import pdb

#%% Global Variables
EQDATAPATH = '/eq/share/lix/data'

#%% Class of GlobalVars
class gvars():
    '''
    All feilds must be in STRING format!
    '''
    # Trade Dates
    StartDate = '20060101'
    TradeDateFile = os.path.join(EQDATAPATH,'tmp/trade.date.txt')
    # Ids
    IdFile = os.path.join(EQDATAPATH,'tmp/ids.txt')
    IdComponets = ['AShareIndices','AShareStocks']
    # Base ini
    BaseIni = os.path.join(EQDATAPATH,'ini/cn.eq.base.ini')

#%% Class of Instruments
class Instruments():
    pass

class AShareIndices(Instruments):
    tickers = ['csi300',
               'csi500',
               'sse50',
               'csi800'] 
    tickers_sh = ['000300',
                  '000905',
                  '000016',
                  '000906']
#%% Functions
def all_instruments()->list:
    try:
        with open(gvars.IdFile,'r') as f:instruments = f.read().splitlines()
    except:
        instruments = []
    return instruments

def all_ashare_index_instruments_sh()->dict:
    instruments_db = AShareIndices().tickers
    instruments_db_sh = AShareIndices().tickers_sh
    instruments_all = all_instruments()
    instruments = {id:id_sh for id,id_sh in zip(instruments_db,instruments_db_sh) if id in instruments_all}
    return instruments

def get_ids_rq():
    # sh:'XSHG',sz:'XSHE'
    d_ids = {''}
    ids = all_instruments()
    ids_index_dict = all_ashare_index_instruments_sh()
    ids2 = [ids_index_dict[id] if id in ids_index_dict else id for id in ids]
    ids_market = instruments_market(ids,sh='.XSHG',sz='.XSHE',idx='.XSHG')
    ids_rq = [id+mkt for id,mkt in zip(ids2,ids_market)]
    return ids_rq

def yyyymmdd2yyyy_mm_dd(yyyymmdd):
    return '-'.join([yyyymmdd[:4],yyyymmdd[4:6],yyyymmdd[-2:]])

def chunks(arr,size):
    sindex = np.arange(0,len(arr),size)
    eindex = np.append(sindex[1:],len(arr)-1)
    return [arr[si:ei] for si,ei in zip(sindex,eindex)]

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

#---------------------- Date ----------------------
def generate_trade_date_file():
    sql = "SELECT \
           TRADE_DAYS \
           FROM \
           WINDDF.ASHARECALENDAR \
           WHERE \
           S_INFO_EXCHMARKET = 'SSE' AND \
           TRADE_DAYS >= {} \
           ORDER BY \
           TRADE_DAYS ASC".format('10000000')
    conn = ora.connect(gvars.ConnWinddb)
    dts = pd.read_sql(sql,conn)
    dts.to_csv(gvars.TradeDateFile,header = False,index = False)

def all_trade_dates()->list:
    try:
        with open(gvars.TradeDateFile,'r') as f:dates = f.read().splitlines()
    except:
        dates = []
    return dates

def n_all_trade_dates()->int:
    return len(open(gvars.TradeDateFile,'r').readlines())

def global_trade_dates()->list:
    return get_dates(sdate=gvars.StartDate,edate=Ini().find('today'))

def datestr2num(datestr:list)->np.ndarray:
    return np.array([int(dt) for dt in datestr])

def datenum2str(datenum:np.ndarray)->list:
    return [str(int(dt)) for dt in datenum]

def date_offset(date:str,offset:int=-1)->str:
    try:
        dates = datestr2num(all_trade_dates())
        dt = dates[np.where(dates>=int(date))[0][0]+offset]
        return str(dt)
    except: return ''

def get_dates(sdate:str=None,edate:str=None,window:int=None,
              dates:list=None,type:str='[]'):
    if dates is None:
        if (sdate is not None) and (edate is not None):
            dates = datestr2num(all_trade_dates())
            dates = dates[(dates>=int(sdate)) & (dates<=int(edate))]
            dates = datenum2str(dates)
        elif (sdate is None) and (edate is not None):
            sdate = date_offset(edate,offset=-window+1)
            dates = get_dates(sdate=sdate,edate=edate)
        elif (edate is None) and (sdate is not None):
            edate = date_offset(sdate,offset=window-1)
            dates = get_dates(sdate=sdate,edate=edate)
    if '(' in type: del dates[0]
    if ')' in type: del dates[-1]
    return dates

#---------------------- Ids ----------------------
def generate_instruments_file():
    instruments = []
    for cp in gvars.IdComponets:
        instruments += (globals()[cp]().tickers)
    instruments.sort()
    with open(gvars.IdFile, "w", newline="") as f:
        for id in instruments:f.write("%s\n" % id)

def all_instruments()->list:
    try:
        with open(gvars.IdFile,'r') as f:instruments = f.read().splitlines()
    except:
        instruments = []
    return instruments

def n_all_instruments()->int:
    return len(open(gvars.IdFile,'r').readlines())

def instruments_market(instruments:list=None,sh='sh',sz='sz',idx='idx')->list:
    if instruments is None:instruments = all_instruments()
    d = {'00':sz,
         '30':sz,
         '60':sh,
         '68':sh,
         'T0':sh,
         'cs':idx,
         'ss':idx}
    return [d[id[:2]] for id in instruments]

def all_ashare_stock_instruments()->list:
    instruments_db = AShareStocks().tickers
    instruments_all = all_instruments()
    instruments = [id for id in instruments_db if id in instruments_all]
    return instruments

def all_ashare_index_instruments()->list:
    instruments_db = AShareIndices().tickers
    instruments_all = all_instruments()
    instruments = [id for id in instruments_db if id in instruments_all]
    return instruments

def all_ashare_index_instruments_sh()->dict:
    instruments_db = AShareIndices().tickers
    instruments_db_sh = AShareIndices().tickers_sh
    instruments_all = all_instruments()
    instruments = {id:id_sh for id,id_sh in zip(instruments_db,instruments_db_sh) if id in instruments_all}
    return instruments

#%% Class of base
class base_():
    def __init__(self,fini=None):
        if fini is not None:
            self.fini = fini
            self.ini = Ini(fini)

#%% Class of HFData
class HFData(base_):
    def __init__(self,fini):
        super().__init__(fini)
        # From outside
        self.ids = all_instruments()
        # From ini
        self.raw_csv_dir = self.ini.findString('RawCsvDir')
        mkdir(self.raw_csv_dir)
        self.csv_dir = self.ini.findString('CsvDir')
        mkdir(self.csv_dir)
        self.bin_dir = self.ini.findString('BinDir')
        mkdir(self.bin_dir)
        self.start_date = self.ini.findString('StartDate')
        self.end_date = self.ini.findString('EndDate')
        self.trade_dates = get_dates(self.start_date,self.end_date)
        # Get ids in format of ricequant
        self.ids_rq = get_ids_rq()

    def get_price_mb(self,ids,dt,fields):
        return rq.get_price(ids,start_date=yyyymmdd2yyyy_mm_dd(dt),\
            end_date=yyyymmdd2yyyy_mm_dd(dt),fields=fields,frequency=self.freq,\
            adjust_type='none')

    def get_tick(self,ticker,dt):
        return rq.get_price(ticker,start_date=yyyymmdd2yyyy_mm_dd(dt),\
            end_date=yyyymmdd2yyyy_mm_dd(dt),frequency='tick')

#%% Class of MbData
class MbData(HFData):
    def __init__(self,fini,freq):
        super().__init__(fini)
        self.freq = str(freq)+'m'
        self.sub_dir = 'mb'+str(freq)
        self.fields = self.ini.findStringVec('Fields')

    def run_csv(self):
        # Loop self.fields
        for field in self.fields:
            print('**** {} ****'.format(field))
            # Loop self.trade_dates
            for dt in self.trade_dates:
                # csv file name
                dir_field = os.path.join(self.raw_csv_dir,self.sub_dir,field)
                mkdir(dir_field)
                csv_file = os.path.join(dir_field,dt+'.csv')
                if not os.path.exists(csv_file):
                    # Get mb data in form of pd.DataFrame
                    d_raw = pd.DataFrame(self.get_price_mb(self.ids_rq,dt,field),columns=self.ids_rq)
                    d_raw.columns = self.ids
                    d_raw.index = d_raw.index.strftime('%H%M')
                    # Store the DataFrame to the csv
                    d_raw.to_csv(csv_file)
                print('{}|{}'.format(field,dt))

#%% Class of TickData
class TickData(HFData):
    def __init__(self,fini):
        super().__init__(fini)
        self.freq = 'tick'
        self.sub_dir = 'tick'

    def get_raw_csv(self):
        # Loop self.trade_dates
        for dt in self.trade_dates:
            dir_field = os.path.join(self.raw_csv_dir,self.sub_dir,dt)
            mkdir(dir_field)
            for ticker,instr in zip(self.ids_rq,all_instruments()):
                # csv file name
                csv_file = os.path.join(dir_field,instr+'.csv')
                if not os.path.exists(csv_file):
                    d_raw = self.get_tick(ticker,dt)
                    if d_raw is not None:
                        d_raw.index = d_raw.index.strftime('%H%M%S')
                        d_raw['trading_date'] = d_raw['trading_date'].dt.strftime('%Y%m%d')
                        # Store the DataFrame to the csv
                        d_raw.to_csv(csv_file)
                print('{}|{}'.format(dt,instr))

#%% Test Codes
if __name__=='__main__':
    tick = TickData('./ini/mb.ini')
    tick.get_raw_csv()
    #mb = MbData('./ini/mb.ini',freq=5)
    #mb.run_csv()
