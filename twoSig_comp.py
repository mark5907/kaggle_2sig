import pandas as pd
import numpy as np


class TwoSig(object):
    _DATA_FILE = '/Users/leiyao/data/train.h5'
    def __init__(self, fn=None):
        if fn is None:
            fn = self._DATA_FILE
        df = pd.read_hdf(fn)
        df['yc'] = df.groupby('id')['y'].transform(lambda x: x.cumsum())

        inst_ids = df['id'].unique().tolist()
        feat_d = [c for c in df if c.startswith('derived_')]
        feat_f = [c for c in df if c.startswith('fundamental_')]
        feat_t = [c for c in df if c.startswith('technical_')]

        self._df = df
        self._inst_ids = inst_ids
        self._feat_d = feat_d
        self._feat_f = feat_f
        self._feat_t = feat_t

    def _add_enter(self):
        df = self._df
        df['just_enter'] = False

        
