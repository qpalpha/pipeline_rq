# -*- coding: utf-8 -*-
"""
# Author: Li Xiang@CICC.EQ
# Created Time : Mon 22 Jul 2019 01:26:03 PM CST
# File Name: test.py
# Description:
"""
#%% Import Part
import os,sys
import time
import rqdatac as rq
import pymysql
rq.init()
from qp import *
from qpc import *
import warnings
warnings.filterwarnings("ignore")
import pdb

#%% Constant variables
NEW_RULE_DATE_SH = 20180820

#%% Functions
def min_bar(freq):
    bar1 = np.array(pd.date_range(start='09:30:00',end='11:30:00',freq=freq+'min')\
        .strftime('%H%M').astype(int))
    bar2 = np.array(pd.date_range(start='13:00:00',end='15:00:00',freq=freq+'min')\
        .strftime('%H%M').astype(int))
    return np.hstack([bar1[1:],bar2[1:]])

def chunks(arr,size):
    sindex = np.arange(0,len(arr),size)
    eindex = np.append(sindex[1:],len(arr)-1)
    return [arr[si:ei] for si,ei in zip(sindex,eindex)]

def cumdiff(sr):
    return sr.diff().fillna(sr)

def df1_sub_df2(df1,df2):
    df1_tuple = df1.apply(tuple,1)
    df2_tuple = df2.apply(tuple,1).tolist()
    idx = [ii for ii,df in df1_tuple.iteritems() if df not in df2_tuple]
    return df1.loc[idx]

#%% Class of Snapshot
class Snapshot():
    def __init__(self):
        self.ids = all_ids()
        self.ids_rq = rq2qp_ids().values.tolist()
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
        self.catch_result = mat,tickers,data_list[0].datetime.strftime('%Y%m%d.%H%M%S')
        return mat,tickers,dtime

    def save(self,dirs:str=None):
        self.catch()
        mat,tickers,dtime = self.catch_result
        if dirs is None:
            dir_ss = './snapshot'
            mkdir(dir_ss)
            file = os.path.join(dir_ss,dtime+'.csv')
        else:
            file = os.path.join(dirs,dtime+'.csv')
        df = pd.DataFrame(mat,index=tickers,columns=self.snapshot_fields)
        df.to_csv(file)

#%% Class of Tickers
class Tickers(base_):
    def __init__(self,fini):
        super().__init__(fini)
        self.ticker_types = self.ini.findStringVec('TickerTypes')
        self.ticker_dir = self.ini.findString('TickerDir')
        mkdir(self.ticker_dir)
        self.id_dir = self.ini.findString('IdDir')
        mkdir(self.id_dir)
        self.stock_diff_csv = self.ini.findString('StockDiffCsv')

    def update(self):
        self.tickers_dict = {tp:rq.all_instruments(type=tp) for tp in self.ticker_types}

    def update_and_save(self):
        self.update()
        for tp,df in self.tickers_dict.items():
            df.to_csv(os.path.join(self.ticker_dir,tp+'.csv'),encoding='utf_8_sig',\
                    index=False)

    def diff2csv(self):
        # Read StockDiffCsv
        if os.path.exists(self.stock_diff_csv):
            old_diff = pd.read_csv(self.stock_diff_csv,dtype='str').fillna(np.nan)
            os.system('cp {0} {0}.{1}'.format(self.stock_diff_csv,today()))
        else:
            old_diff = pd.DataFrame(columns=['S_INFO_WINDCODE','BEGINDATE','ENDDATE',\
                    'S_INFO_NAME','id','de_listed_date','listed_date','id_rq'])
        # Read ids.txt
        qp_ids = all_ids_types_pd()
        qp_ids = qp_ids[qp_ids=='stock'].index.tolist()
        type_dict = rq_types_mapping(['stock'])
        # Read ticker file of rq
        df = rq_raw_ids_df(type_dict['stock'])[['de_listed_date','listed_date','symbol']]
        df['id_rq'] = df.index
        df.index = [id[:6] for id in df.index]
        df['de_listed_date']= df['de_listed_date'].str.replace('-','')
        df['listed_date']= df['listed_date'].str.replace('-','')
        # Difference between ids.txt and ids from rq
        ids_diff = set(df.index.tolist())-set(qp_ids)
        diff = df.loc[ids_diff,:]
        diff = diff[diff['listed_date']!='29991231']
        new_diff = pd.concat([self._get_missed_ticker_(dd) for dd in diff.iterrows()])\
                .fillna(np.nan)
        new_diff.sort_values(by=new_diff.columns.tolist(),inplace=True)
        new_diff.reset_index(drop=True,inplace=True)
        # Save csv
        new_diff.to_csv(self.stock_diff_csv,index=False)
        # Compare old_diff with new_diff
        new_not_old = df1_sub_df2(new_diff,old_diff)
        # Report if new_not_old is not empty
        if len(new_not_old)>0:
            raise Exception('New difference of ids occured!\n{}'.format(new_not_old)) 
        else:
            os.system('rm {0}.{1}'.format(self.stock_diff_csv,today()))

    def _get_missed_ticker_(self,data):
        id = data[0]
        sql = '''
        SELECT
            S_INFO_WINDCODE,BEGINDATE,ENDDATE,S_INFO_NAME
        FROM
            WINDDF.ASHAREPREVIOUSNAME 
        WHERE
            S_INFO_NAME = '{}';
        '''.format(data[1]['symbol'])
        conn = pymysql.connect(**gvars.WinddfInfo)
        data_wind = pd.read_sql(sql,conn)
        conn.close()
        data_wind['id'] = id
        for ii in data[1].index:
            data_wind[ii] = data[1][ii]
        data_wind.drop(['symbol'],inplace=True,axis=1)
        return data_wind

#%% Class of HFData
class HFData(base_):
    def __init__(self,fini):
        super().__init__(fini)
        # From outside
        self.ids = all_ids()
        self.ids_types = all_ids_types_pd()
        self.types_mapping = rq_types_mapping()
        # From ini
        self.raw_dir = self.ini.findString('RawDir')
        mkdir(self.raw_dir)
        self.csv_dir = self.ini.findString('CsvDir')
        mkdir(self.csv_dir)
        self.bin_dir = self.ini.findString('BinDir')
        mkdir(self.bin_dir)
        self.mb_fields = self.ini.findStringVec('MbFields')
        self.mb_feilds_rename_dict = {'last':'tp',
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

    def get_price_mb(self,ids,dt,fields):
        return rq.get_price(ids,start_date=yyyymmdd2yyyy_mm_dd(dt),\
            end_date=yyyymmdd2yyyy_mm_dd(dt),fields=fields,frequency=self.freq,\
            adjust_type='none')

    def get_tick(self,ticker,dt):
        return rq.get_price(ticker,start_date=yyyymmdd2yyyy_mm_dd(dt),\
            end_date=yyyymmdd2yyyy_mm_dd(dt),frequency='tick')

#%% Class of TickData
class TickData(HFData):
    def __init__(self,fini):
        super().__init__(fini)
        self.freq = 'tick'
        self.sub_dir = 'tick'
        self.tick_file_fields = self.ini.findStringVec('TickFileFields')
        tkr = Tickers(fini)
        tkr.update()
        self.tickers_dict = tkr.tickers_dict
        self.retry_times = self.ini.findInt('RawTickRetryTimes')

    def get_raw_csv(self,sdate=None,edate=None,type='CS'):
        if sdate is None: sdate = self.start_date
        if edate is None: edate = self.end_date
        trade_dates = get_dates(sdate,edate)
        # Get ids and ids_rq
        ids = self._get_ids_(type)
        # Loop trade_dates
        for dt in trade_dates:
            dir_dt = os.path.join(self.raw_dir,self.sub_dir,type,dt)
            mkdir(dir_dt)
            for ticker in ids:
                # File name
                csv_file = ticker+'.csv'
                csv_file_abs = os.path.join(dir_dt,ticker+'.csv')
                gz_file = os.path.join(dir_dt,ticker+'.tgz')
                if not os.path.exists(gz_file):
                    i = 0
                    while i<=self.retry_times:
                        try:
                            d_raw = self.get_tick(ticker,dt)
                            break
                        except:
                            d_raw = None
                        finally:
                            i += 1
                        time.sleep(3)
                    if d_raw is not None:
                        d_raw.index = d_raw.index.strftime('%H%M%S')
                        d_raw['trading_date'] = d_raw['trading_date'].dt.strftime('%Y%m%d')
                        # Store the DataFrame to the csv
                        d_raw.to_csv(csv_file_abs)
                        # Tar csv
                        os.system('cd {};tar zcPf {} {}'.format(dir_dt,gz_file,csv_file))
                        # Delete csv
                        os.system('rm {}'.format(csv_file_abs))
                        print('{}|{}|{}'.format(dt,ticker,i))

    def _get_ids_(self,type):
        df = self.tickers_dict[type]
        if type in ['CS','ETF','INDX','Future','Option']:
            ids = df.order_book_id.sort_values().tolist()
        return ids

    def tick2mb1(self,sdate=None,edate=None):
        # Minute-bar time
        mb1_time = min_bar('1')
        mb1_time_aa = np.hstack([930,mb1_time])
        # Trade dates
        if sdate is None: sdate = self.start_date
        if edate is None: edate = self.end_date
        trade_dates = get_dates(sdate,edate)
        # Loop trade_dates
        for dt in trade_dates:
            t1 = time.time()
            instr_market = ids_market(self.ids)
            # Current rq ticker mapping
            self.ids_pd = rq2qp_ids(dt)
            # Init oaa and caa dataframe
            #   oaa : open aggregate auction
            #   caa : close aggregate auction
            oaa_df = pd.DataFrame(index=self.ids,columns=self.tick_file_fields)
            caa_df = pd.DataFrame(index=self.ids,columns=self.tick_file_fields)
            for ii,((instr_qp,instr),(_,type),mkt) in enumerate(zip(self.ids_pd.items(),self.ids_types.items(),instr_market)):
                rq_type = self.types_mapping[type]
                # If is old shanghai stocks
                is_old_sh = (int(dt)<NEW_RULE_DATE_SH) and (mkt=='sh')
                # gz file name
                gz_file = os.path.join(self.raw_dir,self.sub_dir,rq_type,dt,instr+'.tgz')
                if os.path.exists(gz_file):
                    t01 = time.time()
                    # Modify data_tick for later calculation
                    try:
                        data_tick = read_tick_data_file(gz_file).rename(columns=self.mb_feilds_rename_dict)
                    except:
                        continue
                    # Add time
                    data_tick = self._add_time_(data_tick,is_old_sh)
                    # Split by time
                    oaa,data_tick,caa = self._split_by_time_(data_tick)
                    # Open-aa
                    if len(oaa)>0:oaa_df.loc[instr_qp,:] = oaa
                    if len(caa)>0:caa_df.loc[instr_qp,:] = caa
                    # Prep-calculation
                    data_tick = self._prep_cal_(data_tick)
                    # ------------------ Calculation ------------------ 
                    # 1.open
                    open = data_tick.groupby('time')['tp'].first()
                    # 2.ap,bp,av,bv,tp
                    data1 = data_tick.groupby('time').last()
                    tp = data1['tp']
                    ap1,ap2,ap3,ap4,ap5 = data1['ap1'],data1['ap2'],data1['ap3'],data1['ap4'],data1['ap5']
                    bp1,bp2,bp3,bp4,bp5 = data1['bp1'],data1['bp2'],data1['bp3'],data1['bp4'],data1['bp5']
                    av1,av2,av3,av4,av5 = data1['av1'],data1['av2'],data1['av3'],data1['av4'],data1['av5']    
                    bv1,bv2,bv3,bv4,bv5 = data1['bv1'],data1['bv2'],data1['bv3'],data1['bv4'],data1['bv5'] 
                    # 3.volumes and vwapsums
                    v_list = ['volume','vwapsum','luvolume','ldvolume','luvwapsum','ldvwapsum']
                    data_v = data_tick.groupby('time')[v_list].sum()
                    for v in v_list:
                        exec('{0} = data_v[\'{0}\']'.format(v))
                    vwap = data_v['vwapsum']/data_v['volume']
                    # 4.limitup,limitdown
                    limitup = np.logical_and(data1['ap1'].isnull(),data1['bp1']>0).astype(int)
                    limitdown = np.logical_and(data1['ap1']>0,data1['bp1'].isnull()).astype(int)
                    # 5.mid,lsp,lspp
                    mid = (data1['ap1']+data1['bp1'])/2
                    lsp = data1['ap1']-data1['bp1']
                    lspp = lsp/mid
                    # 6.high,low
                    high_df = data_tick.groupby('time')[['tp','high']].max()
                    high_df.loc[high_df['high'].duplicated(),'high'] = np.nan
                    high = high_df.max(axis=1)
                    low_df = data_tick.groupby('time')[['tp','low']].min()
                    low_df.loc[low_df['low'].duplicated(),'low'] = np.nan
                    low = low_df.min(axis=1)
                    # 7.ntick,sp
                    ntick = data_tick.groupby('time').size()
                    abp_mean = data_tick.groupby('time')['ap1','bp1'].mean()
                    sp = abp_mean['ap1'] - abp_mean['bp1']
                    t02 = time.time()
                    # DataFrame -> csv -> gz
                    for field in self.mb_fields:
                        dir_dest = os.path.join(self.csv_dir,'ashare','mb1',field)
                        mkdir(dir_dest)
                        csv_dest = dt+'.csv'
                        csv_dest_abs = os.path.join(dir_dest,csv_dest)
                        gz_dest = dt+'.tgz'
                        new = eval(field)
                        new.name = field
                        new = new.to_frame()
                        new['ticker'] = instr_qp
                        if ii==0:
                            os.system('rm {} -rf'.format(csv_dest_abs))
                            new.to_csv(csv_dest_abs,mode='a')
                        else:
                            new.to_csv(csv_dest_abs,header=False,mode='a')
                        if (ii==(len(self.ids)-1)):
                            # Tar csv
                            os.system('cd {};tar zcPf {} {}'.format(dir_dest,gz_dest,csv_dest))
                            # Delete csv
                            os.system('rm {}'.format(csv_dest_abs))
                    t03 = time.time()
            # oaa and caa
            oaa_dest_dir = os.path.join(self.csv_dir,'ashare/aa','open')
            mkdir(oaa_dest_dir)
            caa_dest_dir = os.path.join(self.csv_dir,'ashare/aa','close')
            mkdir(caa_dest_dir)
            oaa_dest_csv = os.path.join(oaa_dest_dir,dt+'.csv')
            caa_dest_csv = os.path.join(caa_dest_dir,dt+'.csv')
            oaa_df.to_csv(oaa_dest_csv)
            os.system('cd {};tar zcPf {} {}'.format(oaa_dest_dir,dt+'.tgz',dt+'.csv'))
            os.system('rm {}'.format(oaa_dest_csv))
            caa_df.to_csv(caa_dest_csv)
            os.system('cd {};tar zcPf {} {}'.format(caa_dest_dir,dt+'.tgz',dt+'.csv'))
            os.system('rm {}'.format(caa_dest_csv))
            t2 = time.time()
            print('** {} is done, costs {:.2f}s **'.format(dt,t2-t1))    

    def _add_time_(self,data,is_old_sh):
        index = pd.to_datetime(data.index.astype(str),format='%H%M%S')+pd.DateOffset(minutes=1)
        data['time']= index.strftime('%H%M').astype(int)
        data['time'][data['time']<930] = 925
        if not is_old_sh: 
            data['time'][data['time']>1457] = 1501
        else:
            data['time'][data['time']>1500] = 1500
        return data

    def _split_by_time_(self,data):
        oaa = data[data['time']<930]
        caa = data[data['time']>1500]
        if len(oaa)>0:
            oaa = oaa.iloc[-1,:]
        if len(caa)>0:
            caa = caa.iloc[-1,:]
        intraday = data[(data['time']>=930) & (data['time']<=1500)]
        return oaa,intraday,caa
    
    def _prep_cal_(self,data):
        data['volume'] = cumdiff(data['cumvolume'])
        data['vwapsum'] = cumdiff(data['cumvwapsum'])
        data['lul'] = np.logical_and(data['ap1']==0,data['bp1']>0).astype(int)
        data['ldl'] = np.logical_and(data['ap1']>0,data['bp1']==0).astype(int)
        data['luvolume'] = data['volume']*data['lul']
        data['ldvolume'] = data['volume']*data['ldl']
        data['luvwapsum'] = data['vwapsum']*data['lul']
        data['ldvwapsum'] = data['vwapsum']*data['ldl']
        data['ap1'][data['ap1']==0] = np.nan
        data['bp1'][data['bp1']==0] = np.nan
        return data

#%% Class of MBData
class MBData(HFData):
    def __init__(self,fini,freq='5'):
        super().__init__(fini)
        self.freq = freq
        self.sub_dir = 'mb'+freq
        self.MB = min_bar(freq)

    def to_csv(self,sdate=None,edate=None):
        if sdate is None: sdate = self.start_date
        if edate is None: edate = self.end_date
        trade_dates = get_dates(sdate,edate)
        # Loop trade_dates
        for dt in trade_dates:
            t1 = time.time()
            dir_mb1 = os.path.join(self.csv_dir,'ashare/mb1')
            dir_mb = os.path.join(self.csv_dir,'ashare',self.sub_dir)
            self.is_before_new_rule = (int(dt)<NEW_RULE_DATE_SH)
            # Loop mb_fields
            #for field in ['vwap','vwapsum']:
            for field in self.mb_fields:
                # Read mb1 file
                mb1 = read_mb1_data(dt,field)
                # Fill nans
                mb1 = self._fillna_(field,mb1)
                # Change time
                mb1 = self._change_time_(mb1)
                # Do the calculation
                if field in ['sp']:
                    ntick1 = read_mb1_data(dt,'ntick')
                    ntick1 = self._fillna_('ntick',ntick1)
                    ntick1 = self._change_time_(ntick1)
                    sum_sp1 = ntick1*mb1
                    ntick = ntick1.groupby(ntick1.index).sum()
                    sum_sp = sum_sp1.groupby(sum_sp1.index).sum()
                    mb = sum_sp/ntick
                elif field in ['vwap']:
                    vwapsum1 = read_mb1_data(dt,'vwapsum')
                    vwapsum1 = self._fillna_('vwapsum',vwapsum1)
                    vwapsum1 = self._change_time_(vwapsum1)
                    vwapsum = vwapsum1.groupby(vwapsum1.index).sum()
                    volume1 = read_mb1_data(dt,'volume')
                    volume1 = self._fillna_('volume',volume1)
                    volume1 = self._change_time_(volume1)
                    volume = volume1.groupby(volume1.index).sum()
                    mb = vwapsum/volume
                elif field in ['vwapsum','volume','ntick']:
                    mb = eval(field)
                else:
                    mb = mb1.groupby(mb1.index).apply(lambda d:self._compound_(field,d))
                # Save csv
                dir_dest = os.path.join(dir_mb,field)
                mkdir(dir_dest)
                csv_mb = os.path.join(dir_dest,dt+'.csv')
                mb.to_csv(csv_mb)
            t2 = time.time()
            # Print log
            print('mb{0}|{1}|{2:.2f}s'.format(self.freq,dt,t2-t1))

    def to_bin(self,sdate='20100101'):
        pdb.set_trace()
        pass 
            

    def _change_time_(self,data):
        data.index = data.index.map(lambda t:self.MB[np.where(t<=self.MB)[0][0]])
        return data
        
    def _fillna_(self,field,data):
        if field in ['luvolume','ldvolume','luvwapsum','ldvwapsum','ntick',\
                'volume','vwapsum']:
            data.fillna(0,inplace=True)
        else:
            data.fillna(method='ffill',inplace=True)
        return data

    def _compound_(self,field,data):
        if field in ['open']:
            cdata = data.iloc[0,:]
        elif field in ['high']:
            cdata = data.max()
        elif field in ['low']:
            cdata = data.min()
        elif field in ['luvolume','ldvolume','luvwapsum','ldvwapsum']:
            cdata = data.sum()
        else:
            cdata = data.iloc[-1,:]
        return cdata

#%% Test Codes
if __name__=='__main__':
    #print(get_ids_rq())
    #tickers = Tickers('./ini/mb.ini')
    #tickers.diff2csv()
    #tick = TickData('./ini/mb1.history.ini')
    #tick.get_raw_csv(sdate='20100101',edate='20190903')
    #tick.tick2mb1()
    mb = MBData('./ini/mb.ini','5')
    #mb.to_bin()
    mb.to_csv()
    #mb = MBData('./ini/mb.ini','15')
    #mb.to_csv()
    #mb = MBData('./ini/mb.ini','30')
    #mb.to_csv()
