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
from qpc import *
import warnings
warnings.filterwarnings("ignore")
import pdb

#%% Constant variables
MINS_MB1 = np.hstack([925,np.array(pd.date_range(start='09:31:00',end='15:00:00',freq='min')\
        .strftime('%H%M').astype(int))])

NEW_RULE_DATE_SH = 20180820

#%% Functions
def ids_market(instruments:list=None,sh='sh',sz='sz',idx='idx')->list:
    if instruments is None:instruments = all_instruments()
    d = {'00':sz,
         '30':sz,
         '60':sh,
         '68':sh,
         'T0':sh,
         'cs':idx,
         'ss':idx}
    return [d[id[:2]] for id in instruments]

def get_ids_rq():
    # sh:'XSHG',sz:'XSHE'
    d_ids = {''}
    ids = all_ids()
    ids_index_dict = all_ashare_index_ids_sh()
    ids2 = [ids_index_dict[id] if id in ids_index_dict else id for id in ids]
    ids_mkt = ids_market(ids,sh='.XSHG',sz='.XSHE',idx='.XSHG')
    ids_rq = [id+mkt for id,mkt in zip(ids2,ids_mkt)]
    return ids_rq

def chunks(arr,size):
    sindex = np.arange(0,len(arr),size)
    eindex = np.append(sindex[1:],len(arr)-1)
    return [arr[si:ei] for si,ei in zip(sindex,eindex)]

def cumdiff(sr):
    return sr.diff().fillna(sr)

#%% Class of HFData
class HFData(base_):
    def __init__(self,fini):
        super().__init__(fini)
        # From outside
        self.ids = all_ids()
        # From ini
        self.raw_dir = self.ini.findString('RawDir')
        mkdir(self.raw_dir)
        self.csv_dir = self.ini.findString('CsvDir')
        mkdir(self.csv_dir)
        self.bin_dir = self.ini.findString('BinDir')
        mkdir(self.bin_dir)
        self.mb_fields = self.ini.findStringVec('MbFields')
        self.mb_fields_dict = {'last':'tp',
                               'volume':'cumvolume',
                               'total_turnover':'cumvwapsum',
                               'a1':'ap1',
                               'a2':'ap2',
                               'a3':'ap3',
                               'a4':'ap4',
                               'a5':'ap5',
                               'b1':'bp1',
                               'b2':'bp2',
                               'b3':'bp3',
                               'b4':'bp4',
                               'b5':'bp5',
                               'a1_v':'av1',
                               'a2_v':'av2',
                               'a3_v':'av3',
                               'a4_v':'av4',
                               'a5_v':'av5',
                               'b1_v':'bv1',
                               'b2_v':'bv2',
                               'b3_v':'bv3',
                               'b4_v':'bv4',
                               'b5_v':'bv5'}
        self.snapshot_fields = self.ini.findStringVec('SnapShotFields')
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

#%% Class of Snapshot
class Snapshot():
    def __init__(self):
        self.ids = all_ids()
        self.ids_rq = get_ids_rq()
        self.snapshot_fields = ['av1','av2','av3','av4','av5','a1','a2','a3','a4','a5',\
                                'bv1','bv2','bv3','bv4','bv5','b1','b2','b3','b4','b5',\
                                'high','last','low','open','open_interest','prev_close',\
                                'prev_settlement','total_turnover','volume','time','date']

    def _get_data(self,data):
        if not isinstance(data.ask_vols,list):
            abv = [np.nan]*20
        else:
            abv = data.ask_vols+data.asks+data.bid_vols+data.bids
        dt = data.datetime
        return abv+[data.high,data.last,data.low,data.open,data.open_interest,\
            data.prev_close,data.prev_settlement,data.total_turnover,data.volume,\
            float(dt.strftime('%H%M%S'))]

    def catch(self):
        # Ticker
        tickers_dict = {tk_rq:tk for tk_rq,tk in zip(self.ids_rq,self.ids)}
        # Get current_snapshot
        data_list = rq.current_snapshot(self.ids_rq)
        # Time and date vectors
        dtime = data_list[0].datetime.strftime('%Y%m%d')
        l_ = len(data_list)
        dvec = np.full([l_,1],dtime)
        # Stack data and tickers
        mat = np.vstack([self._get_data(data) for data in data_list])
        mat = np.hstack([mat,dvec]).astype(float)
        tickers = [tickers_dict[data._order_book_id] for data in data_list]
        # Return
        return mat,tickers,dtime

    def save(self,file:str=None):
        t1 = time.time()
        mat,tickers,dtime = self.catch()
        if file is None:
            dir_ss = './snapshot'
            mkdir(dir_ss)
            file = os.path.join(dir_ss,dtime+'.csv')
        df = pd.DataFrame(mat,index=tickers,columns=self.snapshot_fields)
        df.to_csv(file)
        t2 = time.time()
        print(t2-t1)

#%% Class of TickData
class TickData(HFData):
    def __init__(self,fini):
        super().__init__(fini)
        self.freq = 'tick'
        self.sub_dir = 'tick'

    def get_raw_csv(self,sdate=None,edate=None):
        if sdate is None: sdate = self.start_date
        if edate is None: edate = self.end_date
        trade_dates = get_dates(sdate,edate)
        # Loop trade_dates
        for dt in trade_dates:
            dir_dt = os.path.join(self.raw_dir,self.sub_dir,dt)
            mkdir(dir_dt)
            for ticker,instr in zip(self.ids_rq,self.ids):
                # File name
                csv_file = os.path.join(dir_dt,instr+'.csv')
                gz_file = os.path.join(dir_dt,instr+'.tar.gz')
                if not os.path.exists(gz_file):
                    d_raw = self.get_tick(ticker,dt)
                    if d_raw is not None:
                        d_raw.index = d_raw.index.strftime('%H%M%S')
                        d_raw['trading_date'] = d_raw['trading_date'].dt.strftime('%Y%m%d')
                        # Store the DataFrame to the csv
                        d_raw.to_csv(csv_file)
                        # Tar csv
                        os.system('tar zcPf {} {}'.format(gz_file,csv_file))
                        # Delete csv
                        os.system('rm {}'.format(csv_file))
                print('{}|{}'.format(dt,instr))

    def tick2mb1(self,sdate=None,edate=None):
        if sdate is None: sdate = self.start_date
        if edate is None: edate = self.end_date
        trade_dates = get_dates(sdate,edate)
        #ids = self.ids[:10]
        ids = self.ids
        # Loop trade_dates
        for dt in trade_dates:
            t1 = time.time()
            dir_dt = os.path.join(self.raw_dir,self.sub_dir,dt)
            for ii,instr in enumerate(ids):
                # csv file name
                gz_file = os.path.join(dir_dt,instr+'.tar.gz')
                if os.path.exists(gz_file):
                    t01 = time.time()
                    # Modify data_tick for later calculation
                    try:
                        data_tick = read_tick_data_file(gz_file).rename(columns=self.mb_fields_dict)
                    except:
                        continue
                    index = pd.to_datetime(data_tick.index.astype(str),format='%H%M%S')+pd.DateOffset(minutes=1)
                    data_tick['time']= index.strftime('%H%M').astype(int)
                    data_tick['time'][data_tick['time']<930] = 925
                    data_tick['volume'] = cumdiff(data_tick['cumvolume'])
                    data_tick['vwapsum'] = cumdiff(data_tick['cumvwapsum'])
                    data_tick['lul'] = np.logical_and(data_tick['ap1']==0,data_tick['bp1']>0).astype(int)
                    data_tick['ldl'] = np.logical_and(data_tick['ap1']>0,data_tick['bp1']==0).astype(int)
                    data_tick['luvolume'] = data_tick['volume']*data_tick['lul']
                    data_tick['ldvolume'] = data_tick['volume']*data_tick['ldl']
                    data_tick['luvwapsum'] = data_tick['vwapsum']*data_tick['lul']
                    data_tick['ldvwapsum'] = data_tick['vwapsum']*data_tick['ldl']
                    data_tick['ap1'][data_tick['ap1']==0] = np.nan
                    data_tick['bp1'][data_tick['bp1']==0] = np.nan
                    # If is old shanghai stocks
                    instr_market = ids_market([instr])[0]
                    is_old_sh = (int(dt)<NEW_RULE_DATE_SH) and (instr_market=='sh')
                    if not is_old_sh: 
                        data_tick['time'][data_tick['time']>1457] = 1500
                    else:
                        data_tick['time'][data_tick['time']>1500] = 1500
                    # tick -> mb1
                    # open
                    open = data_tick.groupby('time')['tp'].first()
                    # ap,bp,av,bv,tp
                    data1 = data_tick.groupby('time').last()
                    tp = data1['tp']
                    ap1,ap2,ap3,ap4,ap5 = data1['ap1'],data1['ap2'],data1['ap3'],data1['ap4'],data1['ap5']
                    bp1,bp2,bp3,bp4,bp5 = data1['bp1'],data1['bp2'],data1['bp3'],data1['bp4'],data1['bp5']
                    av1,av2,av3,av4,av5 = data1['av1'],data1['av2'],data1['av3'],data1['av4'],data1['av5']    
                    bv1,bv2,bv3,bv4,bv5 = data1['bv1'],data1['bv2'],data1['bv3'],data1['bv4'],data1['bv5'] 
                    # volumes and vwapsums
                    v_list = ['volume','vwapsum','luvolume','ldvolume','luvwapsum','ldvwapsum']
                    data_v = data_tick.groupby('time')[v_list].sum()
                    for v in v_list:
                        exec('{0} = data_v[\'{0}\']'.format(v))
                    vwap = data_v['vwapsum']/data_v['volume']
                    # mid,lsp,lspp
                    limitup = np.logical_and(data1['ap1'].isnull(),data1['bp1']>0).astype(int)
                    limitdown = np.logical_and(data1['ap1']>0,data1['bp1'].isnull()).astype(int)
                    mid = (data1['ap1']+data1['bp1'])/2
                    lsp = data1['ap1']-data1['bp1']
                    lspp = lsp/mid
                    # high,low,sp
                    idx_open_to_delete = data_tick.index[data_tick['time']<930][:-1]
                    data_tick.drop(index=idx_open_to_delete,inplace=True)
                    if not is_old_sh:
                        idx_close_to_delete = data_tick.index[data_tick['time']>1457][:-1]
                        data_tick.drop(index=idx_close_to_delete,inplace=True)
                    high = data_tick.groupby('time')['tp'].max() 
                    low = data_tick.groupby('time')['tp'].min() 
                    abp_mean = data_tick.groupby('time')['ap1','bp1'].mean()
                    sp = abp_mean['ap1'] - abp_mean['bp1']
                    t02 = time.time()
                    # DataFrame -> csv -> gz
                    for field in self.mb_fields:
                        dir_dest = os.path.join(self.csv_dir,'mb1',field)
                        mkdir(dir_dest)
                        csv_dest = os.path.join(dir_dest,dt+'.csv')
                        gz_dest = os.path.join(dir_dest,dt+'.tar.gz')
                        new = eval(field)
                        new.name = field
                        new = new.to_frame()
                        new['ticker'] = instr
                        #new = pd.DataFrame(new,columns=['ticker',field])
                        if ii==0:
                            os.system('rm {} -rf'.format(csv_dest))
                            new.to_csv(csv_dest,mode='a')
                        else:
                            new.to_csv(csv_dest,header=False,mode='a')
                        if (ii==(len(ids)-1)):
                            # Tar csv
                            os.system('tar zcPf {} {}'.format(gz_dest,csv_dest))
                            # Delete csv
                            os.system('rm {}'.format(csv_dest))
                    t03 = time.time()
                    #print('  Costs {0:.4f} + {1:.4f} = {2:.4f}s'.format(t02-t01,t03-t02,t03-t01))
                #print('{}|{}'.format(dt,instr))    
            t2 = time.time()
            print('** {} is done, costs {:.2f}s **'.format(dt,t2-t1))    


#%% Test Codes
if __name__=='__main__':
    tick = Snapshot()
    tick.catch()
    
