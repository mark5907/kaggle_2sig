import pandas as pd
import numpy as np


class TwoSig(object):
    _DATA_FILE = '/Users/leiyao/data/kaggle_2sig/train.h5'
    def __init__(self, fn=None):
        if fn is None:
            fn = self._DATA_FILE
        df = pd.read_hdf(fn)
        df['yc'] = df.groupby('id')['y'].transform(lambda x: x.cumsum())
        inst_ids = df['id'].unique().tolist()
        
        newname = {}
        for feat in ['fundamental_', 'derived_', 'technical_']:
            tmp = {c:c.replace(feat, feat[0]+'_') for c in df if c.startswith(feat)}
            newname.update(tmp)

        df.rename(columns=newname, inplace=True)
        self._df = df


    def _add_enter_exit_time(self):
        ef = self._df.pivot(index='timestamp', columns='id', values='y')
        ef = ef.notnull().astype(int).diff()

        dat_enter = []
        dat_exit  = []
        for c in ef:
            enter = ef.loc[ef[c]==1, c].index.tolist()

            exit  = ef.loc[ef[c]==-1, c].index.tolist()
            exit  = [x - 1 for x in exit]

            if len(enter)>0:
                tmp_enter = zip([c]*len(enter), enter)
                dat_enter.append(tmp_enter)

            if len(exit)>0:
                tmp_exit = zip([c]*len(exit), exit)
                dat_exit.append(tmp_exit)

        df_enter = pd.DataFrame(np.concatenate(dat_enter), columns=['id', 'timestamp'])
        df_enter['just_enter'] = True
        df_exit = pd.DataFrame(np.concatenate(dat_exit), columns=['id', 'timestamp'])
        df_exit['exit_next'] = True

        df = self._df
        df = pd.merge(df, df_enter, how='left')
        df = pd.merge(df, df_exit, how='left')

        df['just_enter'] = df['just_enter'].fillna(False)
        df['exit_next']  = df['exit_next'].fillna(False)
        self._df = df
        
         